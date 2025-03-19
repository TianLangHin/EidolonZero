async function helloWorld() {
  const response = await fetch('http://127.0.0.1:5000/')
  const jsonResponse = await response.json()
  document.getElementById('hello').innerHTML = `${jsonResponse.text} and ${jsonResponse.misc}`
}

async function getMoves() {
  const fenString = document.getElementById('fen').value
  const params = new URLSearchParams({ fen: fenString })
  const response = await fetch('http://127.0.0.1:5000/testlegalmoves?' + params.toString())
  const jsonResponse = await response.json()
  document.getElementById('moves').innerHTML = jsonResponse.moves
}
