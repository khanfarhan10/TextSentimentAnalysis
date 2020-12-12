var SpeechRecognition = window.webkitSpeechRecognition;
  
var recognition = new SpeechRecognition();

var input_field = $('#textbox');
var instructions = $('instructions');

var text = '';

recognition.continuous = true;

recognition.onresult = function(event) {

  var current = event.resultIndex;

  var transcript = event.results[current][0].transcript;
 
    text += transcript;
    console.log(transcript);
    input_field.val(text);
  
};

recognition.onstart = function() { 
    console.log("starting");
}

recognition.onspeechend = function() {
    console.log("ending");
}

recognition.onerror = function(event) {
  console.log(event.error);
}

$('#start-btn').on('click', function(e) {
  if (text.length) {
    text += ' ';
  }
  recognition.start();
});

input_field.on('input', function() {
  text = $(this).val();
})