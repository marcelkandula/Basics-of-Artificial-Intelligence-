from connect4 import Connect4

"""
    Functions here should return a scalar value of a current 'position'
    in Connect4 game as seen for player playing with 'token' (one of ['o', 'x']).
"""


def simple_score(position: Connect4, token="x"):
    score = 0
    opponent_token = 'o' if token == 'x' else 'x' 

    if position.game_over:
        if position.wins == token:
            return 10000  
        elif position.wins == opponent_token:
            return -10000  
        else:
            return 0 

    for four in position.iter_fours():
        agent_count = four.count(token)
        opponent_count = four.count(opponent_token)
        if agent_count == 3:
            score += 1
        if opponent_count == 3:
            score -= 1

    return score

def advanced_score(position: Connect4, token="x"):
    """Działanie:

    Użyj wyobraźni i stwórz własną heurystykę oceny pozycji.

    Podpowiedzi:

    - pewnie powinno być to rozwinięcie simple_score
    - metoda position.center_column() - zwraca kolumnę środkową - pewnie nie jest napisana bez powodu
    """
    score = 0
    center_column = position.center_column()
    center_column_tokens = center_column.count(token) 
    opponent_tokens = center_column.count('o' if token == 'x' else 'x')  
    #score += 1 * center_column_tokens - 1 * opponent_tokens  

    for line in position.iter_fours():
        line_score = 0
        agent_tokens = line.count(token)
        opponent_tokens = line.count('o' if token == 'x' else 'x')
        if agent_tokens == 3 and line.count('_') == 1:
            line_score += 1
        elif opponent_tokens == 3 and line.count('_') == 1:
            line_score -= 1

        score += line_score

    return score
