async function helloWorld() {
  const response = await fetch('http://127.0.0.1:5000/')
  const jsonResponse = await response.json()
  document.getElementById('hello').innerHTML = `${jsonResponse.text} and ${jsonResponse.misc}`
}
