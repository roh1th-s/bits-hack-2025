from flask import Flask, request, jsonify
import re
import random

import requests
from insta_scraper import get_instagram_account_data

app = Flask(__name__)

def analyze_instagram_account(username):
    """
    Analyze an Instagram username to determine if it might be a fake account.
    This uses some basic heuristics that could indicate a fake account.
    
    In a real application, you would want to:
    1. Use Instagram's API (with proper authentication)
    2. Analyze more sophisticated signals (followers/following ratio, post frequency, etc.)
    3. Implement ML-based detection
    """
    # These are just example heuristics - in a real app you'd want more sophisticated checks
    red_flags = []
    score = 0  # Higher score = more likely to be fake
    
    # Check for excessive numbers in username
    if sum(c.isdigit() for c in username) > 4:
        red_flags.append("Excessive numbers in username")
        score += 15
    
    # Check for random character patterns
    if re.search(r'[0-9a-zA-Z]{8,}', username):
        if any(c.isdigit() for c in username) and any(c.isalpha() for c in username):
            red_flags.append("Random alphanumeric pattern detected")
            score += 20
    
    # Check for common fake patterns
    if re.search(r'(official|real|verified|authentic|original)$', username, re.IGNORECASE):
        red_flags.append("Uses credibility terms like 'official', 'real', 'verified'")
        score += 25
        
    # Check for repeated characters
    if re.search(r'(.)\1\1\1', username):
        red_flags.append("Contains repeated characters")
        score += 10
    
    # In a real app, you would check many more signals:
    # - Account creation date
    # - Post frequency
    # - Follower to following ratio
    # - Comment patterns
    # - Profile completeness
    
    # Randomize score a bit to simulate other unknown factors
    score += random.randint(-5, 5)
    
    # Determine fake likelihood based on score
    if score > 40:
        fake_likelihood = "High"
    elif score > 20:
        fake_likelihood = "Medium"
    else:
        fake_likelihood = "Low"
    
    return {
        "username": username,
        "fake_likelihood": fake_likelihood,
        "score": score,
        "red_flags": red_flags,
        "disclaimer": "This is a demonstration only. For a real application, you would need Instagram API access and more sophisticated analysis."
    }

@app.route('/check_account', methods=['GET'])
def check_account():
    print('esfsf')
    username = request.args.get('username')
    
    if not username:
        return jsonify({"error": "Please provide an Instagram username as a query parameter"}), 400
    
    data = get_instagram_account_data(username)

    if data is None:
        return jsonify({"error": "Could not fetch Instagram data for this account"}), 400
    print(data)
    res=requests.post("https://5000-01jnecfjebarp3wa2fmvx8m6es.cloudspaces.litng.ai/evaluate_profile", json=data)
    
    return jsonify(res.json())

# default get route

@app.route('/') 
def home():
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=False)
