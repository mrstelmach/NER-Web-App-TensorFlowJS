![alt text](https://github.com/mrstelmach/NER-Web-App-TensorFlowJS/blob/master/web-app/test/webapp-screen.JPG?raw=true)
# NER-Web-App-TensorFlowJS
Named Entity Recognition Web Application powered by Tensorflow.js. It helps to identify Person, Organization, Location or Miscellaneous type of entities within a given text.

## Demo
Live working demo can be accessed under the following link: https://ner-tensorflowjs.glitch.me/web-app/ner-web-app.html.

## Browser support
Application is available in most popular modern browsers supporting JavaScript and pop-ups allowed. Particularly, it has been thoroughly tested in Google Chrome 92.0.4515.107, Firefox 90.0.2 and Microsoft Edge 91.0.864.71. Currently Internet Explorer is not supported.

## Usage
Simply type in any English text and click on <b>Search</b> button to find out what entities are detected. A <i>"The 2020 UEFA European Football Championship was the 16th UEFA European Championship, the quadrennial international men's football championship of Europe organised by the Union of European Football Associations (UEFA). To celebrate the 60th anniversary of the European Championship competition, UEFA president Michel Platini declared that the tournament would be hosted in several nations."</i> text from the header might be used as an example. 
<br><br>Please note that currently sequences up to <b>64</b> tokens are supported (otherwise an alert is raised and the text is automatically truncated). Consider splitting text for longer sequences.

## Development
All development steps are outlined in `ner-web-app-development.pdf` notebook (please use Download button for viewing). In a nutshell, a <b><a href="https://www.clips.uantwerpen.be/conll2003/ner/">CoNLL-2003 dataset</a></b> was used to build a LSTM, CNN & Embedding based deep learning model with <b>Python</b> in <b><a href="https://www.tensorflow.org/">TensorFlow</a></b>, <b><a href="https://keras.io/keras_tuner/">KerasTuner</a></b> and <b><a href=https://www.tensorflow.org/js>TensorFlow.js</a></b> for web application model deployment. Pretrained <b><a href="https://nlp.stanford.edu/projects/glove/">GloVe embeddings</a></b> were also utilised. Essential packages versions are available in `environment/environment.yml` file. All files required to create the web app (.html, .css, .js) are available in `web-app` folder.
