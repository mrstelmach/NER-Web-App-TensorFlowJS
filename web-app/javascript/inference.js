
/**
 * Change display type for given html element with specified id
 * @param  {String}		id 			Id of html element (e.g. button)
 * @param  {String} 	display  	Display type to be applied
 */	
function changeDisplayType (id, display = 'none') {
	document.getElementById(id)
	.style.display = display;
}


/**
 * Load keras model in tfjs format
 */	
async function loadModel() {
	model = await tf.loadLayersModel('https://apple-glowing-porter.glitch.me/model/model.json');
	console.log("Keras model loaded");
	changeDisplayType('spinner', 'none');
	changeDisplayType('clickButton', 'inline-block');
}


/**
 * Highlight string with background color according to the specified highlighter id
 * @param  {String}		string 			Input string
 * @param  {String} 	title  			Value to be displayed in pop up box
 * @param  {String} 	highlighter  	Id of a specific span class (available in style.css)
 */	
function highlightString(string, title, highlighter) {
	return `<span class=${highlighter} style="font-weight:bold" title=${title}>${string}</span>`;
}


/**
 * Return the same text as in input but with entity tags highlighted
 * (utilizes getPrediction function from textFunctions.js)
 * @param  {String}		inputVal 	Text input
 */	
function getNerTags(inputVal) {
    			
	const pred = getPrediction(
	inputVal, wordVocab, charVocab, labels, window.model, 
	sequenceLength = 64, wordLength = 16
	);
	
	let perTags = ["B-PER", "I-PER"];
	let orgTags = ["B-ORG", "I-ORG"];
	let locTags = ["B-LOC", "I-LOC"];
	let miscTags = ["B-MISC", "I-MISC"];
	let output = "";
	let slicedVal = inputVal;
	
	// iterate over tokens and assign colour based on predicted label
	for (let i = 0; i < pred['tokens'].length; i++) {
		let token = pred['tokens'][i];
		let label = pred['labels'][i];
		let tokenIndex = slicedVal.indexOf(token);
		
		if (slicedVal[0] === " ") {
			var separator = " ";
		} else {
			var separator = "";
		}
		
		if (perTags.includes(label)) {
			var newString = highlightString(token, label, 'highlightPer');
		} else if (orgTags.includes(label)) {
			var newString = highlightString(token, label, 'highlightOrg');
		} else if (locTags.includes(label)) {
			var newString = highlightString(token, label, 'highlightLoc');
		} else if (miscTags.includes(label)) {
			var newString = highlightString(token, label, 'highlightMisc');
		} else {
			var newString = token;
		}
		
		slicedVal = slicedVal.trimLeft().slice(token.length);
		output = output.concat(separator, newString);
		
	}
	
	return output;

}
