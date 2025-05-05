const pieceMap = {
  'X': 'X',
  'P': '♙',
  'p': '♟',
  'B': '♗',
  'b': '♝',
  'Q': '♕',
  'q': '♛',
  'K': '♔',
  'k': '♚',
  'R': '♖',
  'r': '♜',
  'N': '♘',
  'n': '♞'
}

async function step1(daMove) {
  const fen = document.getElementById('fen').innerHTML
  const move = daMove
  const params = new URLSearchParams({ fen: fen, move: daMove })
  const response = await fetch('http://127.0.0.1:5000/makemove?' + params.toString())
  const jsonResponse = await response.json()
  document.getElementById('fen').innerHTML = jsonResponse.new_board.fen;
}

async function step2() {
  const fen = document.getElementById('fen').innerHTML
  const params = new URLSearchParams({ fen: fen, invert: true })
  const response = await fetch('http://127.0.0.1:5000/getfoggedstate?' + params.toString())
  const jsonResponse = await response.json()
  let foggedFen = jsonResponse.fen
  let materialCount = jsonResponse.material

  // Update the UI.
  const visible = jsonResponse.visible
  const squares = jsonResponse.squares
  for (let key in squares) {
    const occupancy = !visible[key] ? 'X' : squares[key] === null ? '' : pieceMap[squares[key]]
    document.getElementById('main-' + key).innerHTML = occupancy
  }

  return [foggedFen, materialCount]
}

async function step3(foggedFen, materialCount) {
  const params = new URLSearchParams({ ...materialCount, fen: foggedFen })
  const response = await fetch('http://127.0.0.1:5000/inference?' + params.toString())
  const jsonResponse = await response.json();

  document.getElementById('ai-best-move').innerHTML = 'Best Move: ' + jsonResponse.move

  const squares = jsonResponse.predicted_board.squares
  for (let key in squares) {
    document.getElementById(key).innerHTML = squares[key] === null ? '' : pieceMap[squares[key]]
  }

  return jsonResponse.move
}

async function step4(aiMove) {
  const fen = document.getElementById('fen').innerHTML
  const params = new URLSearchParams({ fen: fen, move: aiMove })
  const response = await fetch('http://127.0.0.1:5000/makemove?' + params.toString())
  const jsonResponse = await response.json()
  document.getElementById('fen').innerHTML = jsonResponse.new_board.fen
}

async function step5() {
  const fen = document.getElementById('fen').innerHTML
  const params = new URLSearchParams({ fen: fen })
  const response = await fetch('http://127.0.0.1:5000/getfoggedstate?' + params.toString())
  const jsonResponse = await response.json()

  // Update the UI.
  const visible = jsonResponse.visible
  const squares = jsonResponse.squares
  for (let key in squares) {
    const occupancy = !visible[key] ? 'X' : squares[key] === null ? '' : pieceMap[squares[key]]
    document.getElementById('main-' + key).innerHTML = occupancy
  }
}

async function makemove(daMove) {
  await step1(daMove)
  const [foggedFen, materialCount] = await step2()
  const aiMove = await step3(foggedFen, materialCount)
  await step4(aiMove)
  await step5()
}

async function selectPiece(div){
  //blue is temporary highlight
  //default/deselected colour = "?", default based on the class, need to get class that ID is present in somehow
  let secondID = null;
  // board = document.getElementsByClassName("chess-board-lil");
  board = document.getElementById("main-chess-board");
  tr = board.getElementsByTagName("tr");
  for(i = 0; i < tr.length; i++){
    td = tr[i].getElementsByTagName("td"); 
    for(ii = 0; ii < td.length; ii++){
      if(td[ii].style.background === "blue"){  
        secondID = td[ii].id; break; 
      }
    } 
  }

  document.getElementById(div.id).style.background = "blue";

  if(secondID!==null){
    //before concatenating to make uci, have to doctor both id's to remove "main-" prefix 
    let secondIDuci = secondID.replace('main-', '');
    let divIDuci = div.id.replace('main-',''); //turns 'main-a8' into 'a8'
    let daMove = "" + secondIDuci + divIDuci; //UCI string  
    makemove(daMove);

    //returning squares to normal background colour
    if(div.className === "light"){
      document.getElementById(div.id).style.background = "#eee"
    } else {
      document.getElementById(div.id).style.background = "#aaa"
    }
    if(document.getElementById(secondID).className === "light"){
      document.getElementById(secondID).style.background = "#eee"
    } else{ 
      document.getElementById(secondID).style.background = "#aaa"
    }
  }
}
