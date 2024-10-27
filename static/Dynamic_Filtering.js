// JavaScript to handle form interactions and dynamic updates

// Function to dynamically update the age label based on the slider's value
function updateAgeLabel(value) {
    const ageRanges = {
        1: "18-30",
        2: "31-50",
        3: "51-65",
        4: "66+"
    };
    document.getElementById('ageLabel').textContent = `Age Range: ${ageRanges[value]}`;
}

// Function to validate form inputs before submission
function validateForm() {
    // You can add more validation rules as needed
    const ageRange = document.getElementById('ageRange').value;
    const weather = document.getElementById('weather').value;
    const collision = document.getElementById('collision').value;
    const experience = document.getElementById('experience').value;
    const cause = document.getElementById('cause').value;

    if (!ageRange || !weather || !collision || !experience || !cause) {
        alert('Please fill out all fields before submitting.');
        return false;
    }

    return true;
}

// Optional: If you want to use AJAX to submit the form instead of a full page reload
function submitFormWithAjax(event) {
    event.preventDefault();  // Prevent the default form submission

    // Only continue if the form is valid
    if (!validateForm()) {
        return;
    }

    // Gather form data
    const formData = {
        ageRange: document.getElementById('ageRange').value,
        weather: document.getElementById('weather').value,
        collision: document.getElementById('collision').value,
        experience: document.getElementById('experience').value,
        cause: document.getElementById('cause').value
    };

    // Perform an AJAX POST request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
    })
    .then(response => response.json())
    .then(data => {
        // Assuming your Flask app returns JSON with the prediction
        if (data.success) {
            displayResult(data.prediction);  // Display prediction result
        } else {
            alert('Error in prediction.');
        }
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

// Function to display the result on the same page (optional, for AJAX usage)
function displayResult(prediction) {
    const resultContainer = document.createElement('div');
    resultContainer.classList.add('result-container');
    resultContainer.innerHTML = `
        <h2>Prediction Result</h2>
        <p>Your accident severity prediction is: <strong>${prediction}</strong></p>
        <a href="/" class="btn btn-primary">Predict Again</a>
    `;

    // Append the result to the body or another container
    document.body.appendChild(resultContainer);
}

// Attach event listeners once the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');

    // If you want to submit via traditional form submission (full page reload)
    form.addEventListener('submit', function (event) {
        return validateForm();  // Just validate, no AJAX
    });

    // Optional: If you want to use AJAX instead of traditional form submission, enable this:
    // form.addEventListener('submit', submitFormWithAjax);
});
