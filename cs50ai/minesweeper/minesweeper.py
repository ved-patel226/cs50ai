import itertools
import random
import copy
from termcolor import colored


class Minesweeper:
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        self.height = height
        self.width = width
        self.mines = set()

        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            # print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    # print("|" + colored("X", "red"), end="")
                    pass
                else:
                    # print("| ", end="")
                    pass
            # print("|")
        # print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        count = 0

        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                if (i, j) == cell:
                    continue

                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence:
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1
        else:
            pass

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)
        else:
            pass


class MinesweeperAI:
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        self.height = height
        self.width = width

        self.moves_made = set()

        self.mines = set()
        self.safes = set()

        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        self.moves_made.add(cell)

        self.mark_safe(cell)

        cells = set()
        count_cpy = copy.deepcopy(count)
        close_cells = self.return_close_cells(cell)
        for cl in close_cells:
            if cl in self.mines:
                count_cpy -= 1
            if cl not in self.mines | self.safes:
                cells.add(cl)

        new_sentence = Sentence(cells, count_cpy)

        if len(new_sentence.cells) > 0:
            self.knowledge.append(new_sentence)
            # print(colored(f"Adding new sentence: {new_sentence}", "green"))

        # print(colored("Printing knowledge:", "blue"))
        for sent in self.knowledge:
            # print(sent)
            pass

        # check sentences for new cells that could be marked as safe or as mine
        self.check_knowledge()
        # print(colored(f"Safe cells: {self.safes - self.moves_made}", "yellow"))
        # print(colored(f"Mine cells: {self.mines}", "red"))
        # print(colored("------------", "blue"))

        self.extra_inference()

    def return_close_cells(self, cell):
        """
        returns cell that are 1 cell away from cell passed in arg
        """
        close_cells = set()
        for rows in range(self.height):
            for columns in range(self.width):
                if (
                    abs(cell[0] - rows) <= 1
                    and abs(cell[1] - columns) <= 1
                    and (rows, columns) != cell
                ):
                    close_cells.add((rows, columns))
        return close_cells

    def check_knowledge(self):
        """
        check knowledge for new safes and mines, updates knowledge if possible
        """
        knowledge_copy = copy.deepcopy(self.knowledge)

        for sentence in knowledge_copy:
            if len(sentence.cells) == 0:
                try:
                    self.knowledge.remove(sentence)
                except ValueError:
                    pass
            # check for possible mines and safes
            mines = sentence.known_mines()
            safes = sentence.known_safes()

            # update knowledge if mine or safe was found
            if mines:
                for mine in mines:
                    # print(colored(f"Marking {mine} as mine", "red"))
                    self.mark_mine(mine)
                    self.check_knowledge()
            if safes:
                for safe in safes:
                    # print(colored(f"Marking {safe} as safe", "green"))
                    self.mark_safe(safe)
                    self.check_knowledge()

    def extra_inference(self):
        """
        update knowledge based on inference
        """
        for sentence1 in self.knowledge:
            for sentence2 in self.knowledge:
                if sentence1.cells.issubset(sentence2.cells):
                    new_cells = sentence2.cells - sentence1.cells
                    new_count = sentence2.count - sentence1.count
                    new_sentence = Sentence(new_cells, new_count)
                    mines = new_sentence.known_mines()
                    safes = new_sentence.known_safes()
                    if mines:
                        for mine in mines:
                            # print(
                            #     colored(f"Used inference to mark mine: {mine}", "red")
                            # )
                            # print(colored(f"FinalSen: {new_sentence}", "blue"))
                            self.mark_mine(mine)

                    if safes:
                        for safe in safes:
                            # print(
                            #     colored(f"Used inference to mark safe: {safe}", "green")
                            # )
                            # print(colored(f"FinalSen: {new_sentence}", "blue"))
                            self.mark_safe(safe)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for i in self.safes - self.moves_made:
            # print(colored(f"Making {i} move", "green"))
            return i

        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """

        maxmoves = self.width * self.height

        while maxmoves > 0:
            maxmoves -= 1

            row = random.randrange(self.height)
            column = random.randrange(self.width)

            if (row, column) not in self.moves_made | self.mines:
                return (row, column)

        return None

    def print(self, num):
        if num == 0:
            # print(colored("No moves left to make.", "red"))
            # print(colored("......", "red"))
            pass

        if num == 1:
            # print(colored("No known safe moves, AI making random move.", "red"))
            # print(colored("......", "red"))
            pass
        if num == 2:
            # print(colored("AI making safe move.", "red"))
            # print(colored("......", "red"))
            pass
