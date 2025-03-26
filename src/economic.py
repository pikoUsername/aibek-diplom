def calculate_optimal_order(forecast, safety_factor=0.1):
    """
    Рассчитывает рекомендуемый объём закупки.
    forecast – прогнозное значение спроса (число).
    safety_factor – процент от прогнозируемого спроса для покрытия неопределенности.
    Возвращает рекомендованный объём закупки.
    """
    additional = forecast * safety_factor
    recommended_order = forecast + additional
    return recommended_order
