def ensemble_predict(title, text):
    # Obtener predicciones de los 3 modelos
    tf_pred = predict_tf(title, text)
    lr_pred = predict_lr(title, text)
    xgb_pred = predict_xgb(title, text)

    # Ponderación basada en precisión de modelos
    weights = {'tf': 0.5, 'lr': 0.3, 'xgb': 0.2}

    # Calcular predicción final
    final_prob = (
            tf_pred['confidence'] * weights['tf'] +
            lr_pred['confidence'] * weights['lr'] +
            xgb_pred['confidence'] * weights['xgb']
    )

    return {
        'isFake': final_prob > 0.6,
        'confidence': final_prob,
        'models': {
            'tf': tf_pred,
            'lr': lr_pred,
            'xgb': xgb_pred
        }
    }