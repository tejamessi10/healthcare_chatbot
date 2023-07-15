from flask import Flask, render_template, request, jsonify, make_response
from chatbot_gui import chatbot_response
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def indexpage():
    if request.method == "POST":
        print(request.form.get('name'))
        return render_template("index2.html")
    return render_template("index2.html")


@app.route("/entry", methods=['POST'])
def entry():
    req = request.get_json()
    print(req)
    res = make_response(
        jsonify({"name": "{}.".format(chatbot_response(req)), "message": "OK"}), 200)
    return res


if __name__ == "__main__":
    app.run(debug=True)
