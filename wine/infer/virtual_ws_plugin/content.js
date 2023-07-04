// Listen for messages from the popup script
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'fetchData') {
    // Extract the input data from the web page (modify this based on your webpage structure)
    var productStockElement = document.getElementById('productStock');
    var priceElement = document.getElementById('price');
    var prodAlcoholPercentElement = document.getElementById('prodAlcoholPercent_percent');

    var inputData = {
      productStock: productStockElement ? productStockElement.textContent : '',
      price: priceElement ? priceElement.textContent : '',
      prodAlcoholPercent_percent: prodAlcoholPercentElement ? prodAlcoholPercentElement.textContent : '',
    };

    // Make a request to your REST API with the input data
    // Modify the URL and request parameters as needed
    fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputData),
    })
      .then(function(response) {
        return response.json();
      })
      .then(function(data) {
        // Send the input and prediction data back to the popup script
        sendResponse({
          inputData: JSON.stringify(inputData),
          outputData: data.prediction,
        });
      })
      .catch(function(error) {
        console.error('Error:', error);
      });

    // Return true to indicate that the sendResponse callback will be called asynchronously
    return true;
  }
});
