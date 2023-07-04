document.addEventListener('DOMContentLoaded', function() {
  // Send a message to the content script to fetch the input and prediction data
  chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
    chrome.tabs.sendMessage(tabs[0].id, { action: 'fetchData' }, function(response) {
      if (response) {
        // Update the input and output data on the popup page
        document.getElementById('inputData').innerText = 'Input: ' + response.inputData;
        document.getElementById('outputData').innerText = 'Prediction: ' + response.outputData;
      }
    });
  });
});
