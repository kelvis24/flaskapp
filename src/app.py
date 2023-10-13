from flask import Flask

verification_codes = {}
verification_codes["123"] = "Sam Husted"

app = Flask(__name__, static_folder='../frontend', static_url_path='/frontend')

def on_verification_success(email):
    # Implement your desired action here
    print(f"Verification successful for {email}. Executing another function.")

# Endpoint for handling verification links


@app.route('/verify/<code>', methods=['GET'])
def verify_email(code):
    if code in verification_codes:
        email = verification_codes[code]
        on_verification_success(email)
        del verification_codes[code]  # Remove the used code
        return "Verification successful. You can close this page."
    else:
        return "Verification code not found or has already been used."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3023)