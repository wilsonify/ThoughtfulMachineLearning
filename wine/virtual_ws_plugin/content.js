// Listen for messages from the popup script
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.action === 'fetchData') {
      var inputData = {
      productStock: productStockElement ? productStockElement.textContent : '',
      price: priceElement ? priceElement.textContent : '',
      prodAlcoholPercent_percent: prodAlcoholPercentElement ? prodAlcoholPercentElement.textContent : '',
    };
    // Extract the input data from the web page
    // Extract the ratings
    var ratingsList = document.querySelectorAll('.wineRatings_listItem');
    for (var i = 0; i < ratingsList.length; i++) {
      var initialsElement = ratingsList[i].querySelector('.wineRatings_initials');
      var ratingValueElement = ratingsList[i].querySelector('.wineRatings_rating');
      if (initialsElement && ratingValueElement) {
        var initials = initialsElement.textContent.trim();
        var ratingValue = ratingValueElement.textContent.trim();
        inputData[`w${i.toString().padStart(5, '0')}`] = { initials: initials, rating: ratingValue };
      }
    }

    // Extract the prodAlcoholPercent_percent
    var percentElement = document.querySelector('.prodAlcoholPercent_percent');
    if (percentElement) {
      inputData[`w${i.toString().padStart(5, '0')}`] = { prodAlcoholPercent_percent: percentElement.textContent.trim() };
    }

    // Extract the prodAlcoholVolume_text
    var volumeElement = document.querySelector('.prodAlcoholVolume_text');
    if (volumeElement) {
      inputData[`w${i.toString().padStart(5, '0')}`] = { prodAlcoholVolume_text: volumeElement.textContent.trim() };
    }

    console.log(inputData)

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
