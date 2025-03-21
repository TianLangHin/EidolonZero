async function helloWorld() {
  const response = await fetch('http://127.0.0.1:5000/')
  const jsonResponse = await response.json()
  document.getElementById('hello').innerHTML = `${jsonResponse.text} and ${jsonResponse.misc}`
}

async function getMoves() {
  const fenString = document.getElementById('fen').value
  const params = new URLSearchParams({ fen: fenString })
  const response = await fetch('http://127.0.0.1:5000/test/legalmoves?' + params.toString())
  const jsonResponse = await response.json()
  document.getElementById('moves').innerHTML = jsonResponse.moves
}

async function loadTensor() {
  const fenString = document.getElementById('fen').value
  const params = new URLSearchParams({ fen: fenString })
  const response = await fetch('http://127.0.0.1:5000/test/fowmovetensor?' + params.toString())
  const jsonResponse = await response.json()
  var table = document.getElementById('tensor-table')
  table.innerHTML = ""
  var topRow = document.createElement('tr')
  for (var stack = 0; stack < 73; stack++) {
    var cell = document.createElement('td')
    cell.innerHTML = stack
    topRow.appendChild(cell)
  }
  table.appendChild(topRow)
  var gridRow = document.createElement('tr')
  for (var stack = 0; stack < 73; stack++) {
    var cell = document.createElement('td')
    for (var rank = 0; rank < 8; rank++) {
      var row = document.createElement('p')
      row.innerHTML = '[' + Array.from(Array(8).keys())
        .map(file => jsonResponse['tensor'][`${stack}-${rank}-${file}`])
        .join(', ') + ']'
      cell.appendChild(row)
    }
    gridRow.appendChild(cell)
  }
  table.appendChild(gridRow)
}
