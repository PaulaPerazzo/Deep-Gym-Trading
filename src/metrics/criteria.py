import numpy as np

class ProfitCriteria:
    def __init__(self, portfolio_values, total_steps, trading_days_per_year=252):
        self.portfolio_values = portfolio_values
        self.total_steps = total_steps
        self.trading_days_per_year = trading_days_per_year


    def anualized_return(self):
        """
        Calculate the anualized return of the portfolio.

        Parameters
        ----------
        portfolio_values : list
            The list of the portfolio values.
        total_steps : int
            The total steps of the simulation.
        trading_days_per_year : int, optional
            The number of trading days per year, by default 252.

        Returns
        -------
        float
            The anualized return.
        """
        # # initial and final portfolio values
        # initial_value = self.portfolio_values[0]
        # final_value = self.portfolio_values[-1]

        # total_return = final_value / initial_value
        # number_of_years = self.total_steps / self.trading_days_per_year

        # arr = (total_return ** (1 / number_of_years)) - 1

        # return arr
        initial_value = self.portfolio_values[0]
        final_value = self.portfolio_values[-1]

        total_return = final_value / initial_value
        number_of_years = self.total_steps / self.trading_days_per_year

        arr = (total_return ** (1 / number_of_years)) - 1

        return arr


class RiskCriteria:
    def __init__(self, returns=None, portfolio_values=None, trading_days_per_year=252):
        self.returns = returns
        self.trading_days_per_year = trading_days_per_year
        self.portfolio_values = portfolio_values

    
    def annualized_volatility(self):
        """
        Calculate the annualized volatility of the portfolio.

        Returns
        -------
        float
            The annualized volatility.
        """

        daily_volatility = np.std(self.returns)
        annualized_volatility = daily_volatility * np.sqrt(self.trading_days_per_year)
        
        return annualized_volatility


    def max_drawdown(self):
        """
        Calculate the maximum drawdown of the portfolio.

        Returns
        -------
        float
            The maximum drawdown.
        """
        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdown = (running_max - self.portfolio_values) / running_max
        max_drawdown = np.max(drawdown)
        
        return max_drawdown
    

class RiskReturnCriteria:
    def __init__(self, returns=None, portfolio_values=None, risk_free_rate=0.0, trading_days_per_year=252):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.portfolio_values = portfolio_values

    
    def sharpe_ratio(self):
        excess_returns = self.returns - self.risk_free_rate / self.trading_days_per_year
        annualized_return = np.mean(excess_returns) * self.trading_days_per_year
        annualized_volatility = np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
        sharpe_ratio = annualized_return / annualized_volatility
        
        return sharpe_ratio
    

    def calculate_calmar_ratio(self):
        initial_value = self.portfolio_values[0]
        final_value = self.portfolio_values[-1]
        total_return = final_value / initial_value
        number_of_years = len(self.portfolio_values) / self.trading_days_per_year
        annualized_return = (total_return ** (1 / number_of_years)) - 1
    
        risk_criteria = RiskCriteria(returns=self.returns, portfolio_values=self.portfolio_values)
        max_drawdown = risk_criteria.max_drawdown()
        calmar_ratio = annualized_return / max_drawdown
        
        return calmar_ratio


    def calculate_sortino_ratio(self):
        excess_returns = self.returns - self.risk_free_rate / self.trading_days_per_year
        downside_returns = excess_returns[excess_returns < 0]
        annualized_return = np.mean(excess_returns) * self.trading_days_per_year
        downside_deviation = np.std(downside_returns) * np.sqrt(self.trading_days_per_year)
        sortino_ratio = annualized_return / downside_deviation
        
        return sortino_ratio
