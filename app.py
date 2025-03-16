from flask import Flask, after_this_request, jsonify, request

app = Flask(__name__)

@app.route('/')
def root():
    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    return jsonify({
        'text': 'Hello world!',
        'misc': 'other stuff'
    })

if __name__ == '__main__':
    app.run()
