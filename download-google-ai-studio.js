const audio = document.querySelector('audio'); // or specify the needed selector

if (audio) {
  const url = audio.src;
  const a = document.createElement('a');
  a.href = url;
  a.download = 'audio.wav';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
} else {
  console.log('Audio element not found');
}