import pandas as pd

def construir_entrada(paquete, valores, genero, educacion):
    if genero not in paquete["generos"] or educacion not in paquete["educacion"]:
        raise ValueError("Categoría no reconocida")
    fila = {}
    for nombre, campo in paquete["campos"].items():
        valor = float(valores[nombre])
        if not campo["min"] <= valor <= campo["max"]:
            raise ValueError(f"{nombre}: valor fuera del rango del entrenamiento")
        fila[nombre] = valor
    fila.update({"Genero_" + g: int(g == genero) for g in paquete["generos"]})
    fila["Cod_Nivel_Educacion"] = paquete["educacion"][educacion]
    if set(fila) != set(paquete["columnas"]):
        raise ValueError("Las variables de entrada no coinciden con el modelo")
    return pd.DataFrame([fila], columns=paquete["columnas"])

def predecir(paquete, entrada):
    escalados = pd.DataFrame(paquete["scaler"].transform(entrada), columns=paquete["columnas"])
    modelo = paquete["modelo"]
    clase = int(modelo.predict(escalados)[0])
    indice = list(modelo.classes_).index(1)
    probabilidad = float(modelo.predict_proba(escalados)[0, indice])
    return clase, probabilidad
