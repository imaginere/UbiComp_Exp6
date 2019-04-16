// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
KNN Classification on Webcam Images with mobileNet. Built with p5.js
=== */
let video;
// Create a KNN classifier
const knnClassifier = ml5.KNNClassifier();
let featureExtractor;

function setup() {
  // Create a featureExtractor that can extract the already learned features from MobileNet
  featureExtractor = ml5.featureExtractor('MobileNet', modelReady);
  noCanvas();
  // Create a video element
  video = createCapture(VIDEO);
  // Append it to the videoContainer DOM element
  video.parent('videoContainer');
  // Create the UI buttons
  createButtons();
}

function modelReady(){
  select('#status').html('FeatureExtractor(mobileNet model) Loaded')
}

// Add the current frame from the video to the classifier
function addExample(label) {
  // Get the features of the input video
  const features = featureExtractor.infer(video);
  // You can also pass in an optional endpoint, defaut to 'conv_preds'
  // const features = featureExtractor.infer(video, 'conv_preds');
  // You can list all the endpoints by calling the following function
  // console.log('All endpoints: ', featureExtractor.mobilenet.endpoints)

  // Add an example with a label to the classifier
  knnClassifier.addExample(features, label);
  updateCounts();
}

// Predict the current frame.
function classify() {
  // Get the total number of labels from knnClassifier
  const numLabels = knnClassifier.getNumLabels();
  if (numLabels <= 0) {
    console.error('There is no examples in any label');
    return;
  }
  // Get the features of the input video
  const features = featureExtractor.infer(video);

  // Use knnClassifier to classify which label do these features belong to
  // You can pass in a callback function `gotResults` to knnClassifier.classify function
  knnClassifier.classify(features, gotResults);
  // You can also pass in an optional K value, K default to 3
  // knnClassifier.classify(features, 3, gotResults);

  // You can also use the following async/await function to call knnClassifier.classify
  // Remember to add `async` before `function predictClass()`
  // const res = await knnClassifier.classify(features);
  // gotResults(null, res);
}

// A util function to create UI buttons
function createButtons() {
  // When the A button is pressed, add the current frame
  // from the video with a label of "Attention" to the classifier
  buttonA = select('#addClassAttention');
  buttonA.mousePressed(function() {
    addExample('Attention');
  });

  // When the B button is pressed, add the current frame
  // from the video with a label of "Good" to the classifier
  buttonB = select('#addClassGood');
  buttonB.mousePressed(function() {
    addExample('Good');
  });

  // When the C button is pressed, add the current frame
  // from the video with a label of "Great" to the classifier
  buttonC = select('#addClassGreat');
  buttonC.mousePressed(function() {
    addExample('Great');
  });

    // When the D button is pressed, add the current frame
  // from the video with a label of "Great" to the classifier
  buttonD = select('#addClassRock');
  buttonD.mousePressed(function() {
    addExample('Rock');
  });

    // When the E button is pressed, add the current frame
  // from the video with a label of "Great" to the classifier
  buttonE = select('#addClassStop');
  buttonE.mousePressed(function() {
    addExample('Stop');
  });

  // Reset buttons
  resetBtnA = select('#resetAttention');
  resetBtnA.mousePressed(function() {
    clearLabel('Attention');
  });
	
  resetBtnB = select('#resetGood');
  resetBtnB.mousePressed(function() {
    clearLabel('Good');
  });
	
  resetBtnC = select('#resetGreat');
  resetBtnC.mousePressed(function() {
    clearLabel('Great');
  });
  
  resetBtnD = select('#resetRock');
  resetBtnD.mousePressed(function() {
    clearLabel('Rock');
  });
  
  resetBtnE = select('#resetStop');
  resetBtnE.mousePressed(function() {
    clearLabel('Stop');
  });

  // Predict button
  buttonPredict = select('#buttonPredict');
  buttonPredict.mousePressed(classify);

  // Clear all classes button
  buttonClearAll = select('#clearAll');
  buttonClearAll.mousePressed(clearAllLabels);

  // Load saved classifier dataset
  buttonSetData = select('#load');
  buttonSetData.mousePressed(loadMyKNN);

  // Get classifier dataset
  buttonGetData = select('#save');
  buttonGetData.mousePressed(saveMyKNN);
}

// Show the results
function gotResults(err, result) {
  // Display any error
  if (err) {
    console.error(err);
  }

  if (result.confidencesByLabel) {
    const confidences = result.confidencesByLabel;
    // result.label is the label that has the highest confidence
    if (result.label) {
      select('#result').html(result.label);
      select('#confidence').html(`${confidences[result.label] * 100} %`);
    }

    select('#confidenceAttention').html(`${confidences['Attention'] ? confidences['Attention'] * 100 : 0} %`);
    select('#confidenceGood').html(`${confidences['Good'] ? confidences['Good'] * 100 : 0} %`);
    select('#confidenceGreat').html(`${confidences['Great'] ? confidences['Great'] * 100 : 0} %`);
    select('#confidenceRock').html(`${confidences['Rock'] ? confidences['Rock'] * 100 : 0} %`);
    select('#confidenceStop').html(`${confidences['Stop'] ? confidences['Stop'] * 100 : 0} %`);
  }

  classify();
}

// Update the example count for each label	
function updateCounts() {
  const counts = knnClassifier.getCountByLabel();

  select('#exampleAttention').html(counts['Attention'] || 0);
  select('#exampleGood').html(counts['Good'] || 0);
  select('#exampleGreat').html(counts['Great'] || 0);
  select('#exampleRock').html(counts['Rock'] || 0);
  select('#exampleStop').html(counts['Stop'] || 0);
}

// Clear the examples in one label
function clearLabel(label) {
  knnClassifier.clearLabel(label);
  updateCounts();
}

// Clear all the examples in all labels
function clearAllLabels() {
  knnClassifier.clearAllLabels();
  updateCounts();
}

// Save dataset as myKNNDataset.json
function saveMyKNN() {
  knnClassifier.save('myKNNDataset');
}

// Load dataset to the classifier
function loadMyKNN() {
  knnClassifier.load('./myKNNDataset.json', updateCounts);
}
