from flask import Flask, jsonify, request
from flask_cors import CORS
from search import SearchEngine
from add_json import DocumentAdder
from preprocessAndLexiconGen import LexiconLoader, TextPreprocessor
from search import InvertedIndexSearcher, DocumentRetriever
from search import QueryProcessor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Fixed CORS configuration

@app.route('/api/search', methods=['GET'])
def search_documents():
    query = request.args.get('q', '')

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    lexicon_loader = LexiconLoader("lexicon_output")
    text_preprocessor = TextPreprocessor()
    query_processor = QueryProcessor(lexicon_loader, text_preprocessor)
    inverted_index_searcher = InvertedIndexSearcher("inverted_index_output", lexicon_loader)
    document_retriever = DocumentRetriever("test.csv")
    search_engine = SearchEngine(inverted_index_searcher, document_retriever, query_processor)

    # Get all results
    results = search_engine.search(query)

    return jsonify({
        "results": results,
        "total_results": len(results)
    })

@app.route('/api/add', methods=['POST'])
def add_document_endpoint():
    data = request.json
    if not data:
        return jsonify({"error": "Document data is required"}), 400
    doc_adder = DocumentAdder()
    success, error = doc_adder.add_document(data)
    if not success:
        return jsonify({"error": error}), 400
    return jsonify({"message": "Document added successfully"})

if __name__ == '__main__':
    app.run(debug=True)