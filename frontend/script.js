
const form = document.getElementById("predictionForm");

const yearInput = document.getElementById("model_year");
const makeSelect = document.getElementById("make");
const modelSelect = document.getElementById("model");
const typeSelect = document.getElementById("vehicle_type");

const predictBtn = document.getElementById("predictBtn");

const loadingMessage = document.getElementById("loadingMessage");
const errorMessage = document.getElementById("errorMessage");
const resultSection = document.getElementById("resultSection");

const predictedRange = document.getElementById("predictedRange");

// New: Unit selector
const unitSelect = document.getElementById("rangeUnit");

let vehicleData = null;

// Store both prediction values
let predictedMiles = null;
let predictedKm = null;


// -----------------------------------
// Load Vehicle Options
// -----------------------------------

async function loadOptions() {

    try {

        // Show dropdown loading message only
        loadingMessage.classList.remove("hidden");
        loadingMessage.textContent = "Loading vehicle details...";

        // Disable inputs until options are loaded
        yearInput.disabled = true;
        makeSelect.disabled = true;
        modelSelect.disabled = true;
        typeSelect.disabled = true;
        predictBtn.disabled = true;

        const response = await fetch("/api/options");

        if (!response.ok) {

            let errorData = {};

            try {
                errorData = await response.json();
            } catch (_) {}

            throw new Error(
                errorData.error || "Unable to load vehicle details."
            );
        }

        const data = await response.json();

        if (
            !Array.isArray(data.makes) ||
            !data.make_models ||
            !Array.isArray(data.vehicle_types)
        ) {
            throw new Error("Invalid vehicle options received from server.");
        }

        vehicleData = data;


        // Load Manufacturers

        makeSelect.innerHTML =
            '<option value="">Select manufacturer</option>';

        data.makes.forEach(make => {

            const option = document.createElement("option");

            option.value = make;
            option.textContent = make;

            makeSelect.appendChild(option);

        });


        // Load Vehicle Types

        typeSelect.innerHTML =
            '<option value="">Select vehicle type</option>';

        data.vehicle_types.forEach(type => {

            const option = document.createElement("option");

            option.value = type;
            option.textContent = type;

            typeSelect.appendChild(option);

        });


        // Set Model Year Range

        yearInput.min = data.min_year;
        yearInput.max = data.max_year;
        yearInput.value = data.max_year;


        // Enable Inputs

        yearInput.disabled = false;
        makeSelect.disabled = false;
        typeSelect.disabled = false;

        // Model remains disabled until manufacturer is selected
        modelSelect.disabled = true;

        predictBtn.disabled = false;

        console.log("Vehicle options loaded successfully.");

    }

    catch (error) {

        console.error("Options loading error:", error);

        showError(
            error.message || "Unable to load vehicle options. Please refresh the page."
        );

    }

    finally {

        loadingMessage.classList.add("hidden");

    }

}


// -----------------------------------
// Load Models Based on Manufacturer
// -----------------------------------

makeSelect.addEventListener("change", function () {

    const selectedMake = this.value;

    modelSelect.innerHTML =
        '<option value="">Select vehicle model</option>';

    modelSelect.disabled = true;

    if (!selectedMake || !vehicleData) {
        return;
    }

    const models = vehicleData.make_models[selectedMake] || [];

    models.forEach(model => {

        const option = document.createElement("option");

        option.value = model;
        option.textContent = model;

        modelSelect.appendChild(option);

    });

    modelSelect.disabled = models.length === 0;

});


// -----------------------------------
// Change Displayed Range Unit
// -----------------------------------

function updateRangeDisplay() {

    if (predictedMiles === null || predictedKm === null) {
        return;
    }

    if (unitSelect.value === "km") {

        predictedRange.textContent = predictedKm.toFixed(2);

    } else {

        predictedRange.textContent = predictedMiles.toFixed(2);

    }

}


// Listen for unit selection changes

unitSelect.addEventListener("change", updateRangeDisplay);


// -----------------------------------
// Prediction
// -----------------------------------

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    hideMessages();

    if (!vehicleData) {

        showError("Vehicle options are not loaded. Please refresh the page.");

        return;
    }

    const inputData = {

        model_year: Number(yearInput.value),

        make: makeSelect.value,

        model: modelSelect.value,

        vehicle_type: typeSelect.value

    };


    // Validate Input

    if (
        !inputData.model_year ||
        !inputData.make ||
        !inputData.model ||
        !inputData.vehicle_type
    ) {

        showError("Please fill in all vehicle details.");

        return;
    }


    // Show Prediction Loading

    predictBtn.disabled = true;

    loadingMessage.textContent = "Analyzing vehicle details...";

    loadingMessage.classList.remove("hidden");

    predictBtn.querySelector("span:first-child").textContent = "Predicting...";


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(inputData)

        });


        const data = await response.json();

        if (!response.ok) {

            throw new Error(data.error || "Prediction failed.");

        }


        // Validate Prediction Response

        if (
            typeof data.predicted_range_miles !== "number" ||
            !Number.isFinite(data.predicted_range_miles) ||
            typeof data.predicted_range_km !== "number" ||
            !Number.isFinite(data.predicted_range_km)
        ) {

            throw new Error("Invalid prediction received from server.");

        }


        // Store Both Predictions

        predictedMiles = data.predicted_range_miles;
        predictedKm = data.predicted_range_km;


        // Reset Unit to Miles After New Prediction

        unitSelect.value = "miles";

        updateRangeDisplay();


        // Display Prediction

        resultSection.classList.remove("hidden");

        resultSection.scrollIntoView({

            behavior: "smooth",

            block: "nearest"

        });

    }

    catch (error) {

        console.error("Prediction error:", error);

        showError(
            error.message || "Prediction failed. Please try again."
        );

    }

    finally {

        loadingMessage.classList.add("hidden");

        predictBtn.disabled = false;

        predictBtn.querySelector("span:first-child").textContent =
            "Predict Electric Range";

    }

});


// -----------------------------------
// Error Handling
// -----------------------------------

function showError(message) {

    errorMessage.textContent = message;

    errorMessage.classList.remove("hidden");

}


function hideMessages() {

    errorMessage.classList.add("hidden");

    resultSection.classList.add("hidden");

}


// -----------------------------------
// Initialize Application
// -----------------------------------

loadOptions();
