// ============================================================
// CROP RECOMMENDATION
// ============================================================

const cropForm = document.getElementById("cropForm");
const result = document.getElementById("result");


// Store crop prediction history
let predictionHistory = [];


cropForm.addEventListener("submit", async function (event) {

    event.preventDefault();


    const data = {

        N: Number(document.getElementById("N").value),

        P: Number(document.getElementById("P").value),

        K: Number(document.getElementById("K").value),

        temperature: Number(
            document.getElementById("temperature").value
        ),

        humidity: Number(
            document.getElementById("humidity").value
        ),

        ph: Number(
            document.getElementById("ph").value
        ),

        rainfall: Number(
            document.getElementById("rainfall").value
        )
    };


    result.textContent = "Predicting...";


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)
        });


        const prediction = await response.json();


        if (response.ok) {

            predictionHistory.push({

                input: data,

                crop: prediction.recommended_crop
            });


            result.innerHTML = `

                <div class="prediction-result">

                    <h2>
                        🌾 Recommended Crop:
                        ${prediction.recommended_crop}
                    </h2>

                    <p>
                        The AI model recommends
                        <strong>${prediction.recommended_crop}</strong>
                        based on the entered soil and environmental
                        conditions.
                    </p>

                </div>

            `;


            displayHistory();

        }

        else {

            result.textContent =
                "Error: " + prediction.error;

        }

    }

    catch (error) {

        result.textContent =
            "Unable to connect to the AI backend.";

        console.error(error);
    }

});


// ============================================================
// CROP PREDICTION HISTORY
// ============================================================

function displayHistory() {

    let historyContainer =
        document.getElementById("history");


    if (!historyContainer) {

        historyContainer =
            document.createElement("div");

        historyContainer.id = "history";

        result.parentNode.appendChild(
            historyContainer
        );
    }


    let historyHTML = `

        <div class="prediction-history">

            <h2>📋 Prediction History</h2>

            <table>

                <thead>

                    <tr>

                        <th>N</th>

                        <th>P</th>

                        <th>K</th>

                        <th>Temperature</th>

                        <th>Humidity</th>

                        <th>pH</th>

                        <th>Rainfall</th>

                        <th>Crop</th>

                    </tr>

                </thead>


                <tbody>
    `;


    predictionHistory.forEach(function (item) {

        historyHTML += `

            <tr>

                <td>${item.input.N}</td>

                <td>${item.input.P}</td>

                <td>${item.input.K}</td>

                <td>${item.input.temperature}</td>

                <td>${item.input.humidity}</td>

                <td>${item.input.ph}</td>

                <td>${item.input.rainfall}</td>

                <td>🌾 ${item.crop}</td>

            </tr>

        `;

    });


    historyHTML += `

                </tbody>

            </table>

        </div>

    `;


    historyContainer.innerHTML =
        historyHTML;
}


// ============================================================
// DISEASE DETECTION
// ============================================================

const diseaseForm =
    document.getElementById("diseaseForm");

const diseaseImage =
    document.getElementById("diseaseImage");

const diseaseResult =
    document.getElementById("diseaseResult");

const imagePreview =
    document.getElementById("imagePreview");


// ============================================================
// IMAGE PREVIEW
// ============================================================

diseaseImage.addEventListener(
    "change",
    function () {

        const file =
            diseaseImage.files[0];


        if (!file) {

            imagePreview.src = "";

            return;
        }


        const imageURL =
            URL.createObjectURL(file);


        imagePreview.src =
            imageURL;

    }
);


// ============================================================
// DISEASE FORM SUBMISSION
// ============================================================

diseaseForm.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();


        const file =
            diseaseImage.files[0];


        if (!file) {

            diseaseResult.innerHTML = `

                <div class="error-message">

                    Please select a plant image.

                </div>

            `;

            return;
        }


        diseaseResult.innerHTML = `

            <div class="loading-message">

                🔍 Analyzing image...

            </div>

        `;


        const formData =
            new FormData();


        formData.append(
            "image",
            file
        );


        try {

            const response =
                await fetch(
                    "/predict-disease",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const prediction =
                await response.json();


            if (response.ok) {

                let warningHTML = "";


                if (prediction.warning) {

                    warningHTML = `

                        <p class="warning">

                            ⚠️ ${prediction.warning}

                        </p>

                    `;

                }


                diseaseResult.innerHTML = `

                    <div class="disease-result">

                        <h2>
                            🔬 Detected:
                            ${prediction.disease}
                        </h2>


                        <p>

                            📊
                            <strong>Confidence:</strong>

                            ${prediction.confidence}%

                        </p>


                        <h3>
                            💡 Suggested Action
                        </h3>


                        <p>
                            ${prediction.suggestion}
                        </p>


                        ${warningHTML}

                    </div>

                `;

            }

            else {

                diseaseResult.innerHTML = `

                    <div class="error-message">

                        Error:
                        ${prediction.error}

                    </div>

                `;

            }

        }

        catch (error) {

            diseaseResult.innerHTML = `

                <div class="error-message">

                    Unable to connect to
                    the AI backend.

                </div>

            `;

            console.error(error);

        }

    }
);