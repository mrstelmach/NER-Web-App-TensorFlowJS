
/**
 * Preprocess text by separating apostrophes and/or punctuation
 */	
class TextPreprocessor {
	
	constructor(separateApostrophes = true, separatePunctuation = true,
	punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~') {
		this.separateApostrophes = separateApostrophes;
		this.separatePunctuation = separatePunctuation;
		this.punctuation = punctuation;
	}
	
	// Insert space before apostrophes indicating possession or used in contractions
	getApostrophesSeparated(string) {
		const eos = this.punctuation + ' ';
		let reg = new RegExp(
		`([a-z])('|'s|'ll|'m|'re|'ve|'d|n't)([${eos}]|$)`, 'ig');
		var string = string.replace(reg, '$1 $2$3');
		return string;
	}
	
	// Separate (with space) punctuation from words
	getPunctuationSeparated(string) {
		let reg1 = new RegExp(`(\\S)([${this.punctuation}])`, 'ig');
		let reg2 = new RegExp(`([${this.punctuation}])(\\S)`, 'ig');
		var string = string.replace(reg1, '$1 $2')
		.replace(reg2, '$1 $2');
		return string;
	}
	
	// Apply separators chosen in constructor
	getText(string) {
		if (this.separateApostrophes) {
			var string = this.getApostrophesSeparated(string);
		}
		if (this.separatePunctuation) {
			var string = this.getPunctuationSeparated(string);
		}
		return string;
	}
}


/**
 * Pad array (right) with a chosen value
 * @param  {Array} 	  	array   	Input array
 * @param  {Number}   	length  	Length of the new array
 * @param  {Number}   	fill    	Value to fill with
 */	
function padArrayRight(array, length, fill = 0) {
	return Object.assign(new Array(length)
	.fill(fill), array.slice(0, length));
}


/**
 * Split string into smaller chunks (word level by default)
 * @param  {String} 	string   		Input string
 * @param  {Boolean}   	lower  	 		Lower-case letters
 * @param  {Boolean}   	charLevel   	Split on character level
 */	
function tokenize(string, lower = false, charLevel = false) {
	if (lower) {
		var string = string.toLowerCase();
	}
	if (charLevel) {
		return string.split('');
	} else {
		return string.split(' ');
	}
}


/**
 * Get value for a given key from dictionary
 * @param  {String}		key   			Key to be searched in dictionary
 * @param  {Object}		dictionary  	Object with key : value pairs
 * @param  {Number}		unknownId   	Value for unknown key
 */	
function getDictIndex(key, dictionary, unknownId = 1) {
	return (dictionary[key] !== undefined) ? dictionary[key] : unknownId;
}


/**
 * Get dictionary value for each key from provided array
 * @param  {Array}		array   		Array with keys
 * @param  {Object}		dictionary  	Object with key : value pairs
 * @param  {Number}		unknownId   	Value for unknown key
 */	
function getArrayDictIndex(array, dictionary, unknownId = 1) {
	return array.map(key => getDictIndex(key, dictionary, unknownId));
}


/**
 * Raise alert if input length is greater than specified value
 * @param  {Number}		lengthLimit		Length limit
 */	
function tooLongAlert(lengthLimit) {
	var tooLongMsg = `Input length longer than ${lengthLimit} tokens will be truncated.`;
	tooLongMsg += "\r\n";
	tooLongMsg += "Consider splitting to shorter sequences.";
	alert(tooLongMsg);
}


/**
 * Get tokens and predicted tags for given text input
 * @param  {String} 	textInput		Text input
 * @param  {Object}   	wordVocab  		Dictionary with word level vocabulary
 * @param  {Object}   	charVocab   	Dictionary with character level vocabulary
 * @param  {Object} 	labels   		Dictionary with labels
 * @param  {Model} 	  	model   		Tfjs converted Keras model
 * @param  {Number} 	sequenceLength  Maximum sequence length
 * @param  {Number} 	wordLength   	Maximum word length
 * @param  {Number} 	unknownId   	Id of unknown token
 * @param  {Number} 	padId   		Id of pad token
 */	
function getPrediction(
textInput, wordVocab, charVocab, labels, model, sequenceLength = 32, 
wordLength = 8, unknownId = 1, padId = 0) {
	
	// apply text preprocessing
	let textPrc = new TextPreprocessor();
	var textInput = textPrc.getText(textInput);
	
	// tokenize on word level
	let wordTokensLower = tokenize(textInput, lower = true);
	let wordTokensID = wordTokensLower
	.map(key => getDictIndex(key, wordVocab, unknownId));
	const wordInput = padArrayRight(wordTokensID, sequenceLength, padId);
	
	// tokenize on character level
	// wordTokens are sliced as they will be also returned in function output
	let wordTokens = tokenize(textInput, lower = false)
	.slice(0, sequenceLength);
	let wordTokensPadded = padArrayRight(wordTokens, sequenceLength, '');
	const charInput = wordTokensPadded
	.map(wrd => tokenize(wrd, lower = false, charLevel = true))
	.map(arr => getArrayDictIndex(arr, charVocab, unknownId))
	.map(arr => padArrayRight(arr, wordLength, padId));
	
	// create word level mask
	let wordTokensLen = wordTokens.length;
	const wordMask = Object.assign(new Array(sequenceLength)
	.fill(padId), Array(wordTokensLen).fill(1));
	
	// convert to tensors and get prediction
	const tfInput = [
	tf.tensor(wordInput).expandDims(),
	tf.tensor(charInput).expandDims(), 
	tf.tensor(wordMask).expandDims()
	];
	const predIds = model.predict(tfInput).squeeze().argMax(1)
	.slice([0], [wordTokensLen]).arraySync();
	
	// add labels mapping 
	const predLbs = predIds.map(id => getDictIndex(id, labels, "UNK"));
	
	// alert for too long sequences
	if (wordTokensLower.length > sequenceLength) {
		tooLongAlert(sequenceLength);
	}
	
	return {'tokens': wordTokens, 'labels': predLbs};
}
