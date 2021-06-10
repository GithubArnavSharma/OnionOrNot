document.getElementById("predict-button").addEventListener("click", function () {
    let headline = document.getElementById('headline');
    let result = document.getElementById('result-text');

    const getPrediction = async (headline) => {
        const response = await fetch('/predict', {
            method: 'POST',
            body: JSON.stringify({'headline':headline}),
            headers: {
                'Content-Type': 'application/json'
            }
        });
        const myJson = await response.json();
        return myJson.result;
    }

    let predtext = getPrediction(headline.value);
    predtext.then(function (response) {
        result.textContent = response;
    });


}, false);
