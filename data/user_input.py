import datetime

def get_user_portfolio():
    """
    Simulates getting user input for their portfolio.
    In a real app, this would be a UI form or API endpoint.
    Returns a list of dictionaries.
    """
    # For now, we will hardcode a test portfolio.
    # Later we can use inputs or args
    portfolio = [
        {
            "ticker": "AAPL",
            "quantity": 50,
            "purchase_date": "2023-01-15 10:30:00"
        },
        {
            "ticker": "MSFT",
            "quantity": 20,
            "purchase_date": "2023-03-10 14:15:00"
        }
    ]
    return portfolio
