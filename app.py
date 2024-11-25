from flask import Flask,jsonify,request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

@app.route('/api/greet', methods=['GET'])
def greet():
    return jsonify({"message": "Hello from Flask backend!"})

@app.route('/api/sum', methods=['POST'])
def calculate_sum():
    data = request.json
    a = data.get('a', 0)
    b = data.get('b', 0)
    return jsonify({"sum": a + b})

if __name__ == '__main__':
    app.run(debug=True)