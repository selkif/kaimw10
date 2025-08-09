from flask import Blueprint, jsonify
from data.data_processing import get_brent_data

api_routes = Blueprint('api', __name__)

@api_routes.route('/api/brent_data', methods=['GET'])
def brent_data():
    data = get_brent_data()
    return jsonify(data)