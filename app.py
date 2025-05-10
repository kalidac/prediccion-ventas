from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

modelo = joblib.load("modelo_cierre_ventas.pkl")
encoder_industria = joblib.load("encoder_industria.pkl")
encoder_ciudad = joblib.load("encoder_ciudad.pkl")

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json()
    try:
        empleados = int(data["empleados"])
        industria = str(data["industria"])
        ciudad = str(data["ciudad"])
        dias_ciclo = int(data["dias_ciclo"])
    except:
        return jsonify({"error": "Datos inválidos"}), 400

    try:
        industria_cod = encoder_industria.transform([industria])[0]
    except:
        industria_cod = encoder_industria.transform(['Desconocido'])[0]

    try:
        ciudad_cod = encoder_ciudad.transform([ciudad])[0]
    except:
        ciudad_cod = encoder_ciudad.transform(['Desconocido'])[0]

    X = pd.DataFrame([{
        "Empleados": empleados,
        "Industria_cod": industria_cod,
        "Ciudad_cod": ciudad_cod,
        "ciclo_ventas_dias": dias_ciclo
    }])
    prob = modelo.predict_proba(X)[0][1] * 100
    return jsonify({"probabilidad_cierre": round(prob, 2)})

@app.route("/", methods=["GET"])
def home():
    return "Modelo de cierre activo"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
