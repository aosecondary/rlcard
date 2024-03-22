import json
import os
import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.games.leducholdemstate import Game
from rlcard.utils import *

import gymnasium as gym
from gymnasium.spaces import Box, Dict

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        }

class LeducholdemEnv(Env):
    ''' Leduc Hold'em Environment
    '''

    def __init__(self, config):
        ''' Initialize the Limitholdem environment
        '''
        self.name = 'leduc-holdem' 
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)

        # Qui si definisce la forma
        self.actions = ['call', 'raise', 'fold', 'check']
        self.limited_actions = ['call','check','allin','fold'] # Queste solo le uniche azioni disponibile nell'ultimo round di ogni serie

        self.total_suits = 2
        self.cards_each_suit = 3
        self.max_round = 2 # +2 ulteriori mi servono per Public e Public+Hand

        # Ovvero quante vuole in un turno si può agire, l'ultimo limita le legal action
        self.max_sequential_actions = 3

        # Qui inizializzo il card_tensor_shape e action_tensor_shape
        # card_tensor: Ch = N round +2, Row = suits, Col = card per suit
        # action_tensor: Ch = N round * Max raise per round, Row: N Player + 2 (sum, legal), Col: Actions/Betting Options
        
        self.card_tensor_shape = (self.max_round+2, self.total_suits, self.cards_each_suit)
        self.action_tensor_shape = (self.max_round*self.max_sequential_actions, self.num_players+2, len(self.actions))
        
        # Determina le dimensioni massime
        self.sum_channel = self.card_tensor_shape[0] + self.action_tensor_shape[0]
        self.max_rows = max(self.card_tensor_shape[1], self.action_tensor_shape[1])
        self.max_cols = max(self.card_tensor_shape[2], self.action_tensor_shape[2])

        self.state_shape = [[(self.sum_channel, self.max_rows, self.max_cols)] for _ in range(self.num_players)]

        self.action_shape = [None for _ in range(self.num_players)]

        with open(os.path.join(rlcard.__path__[0], 'games/leducholdemstate/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()
    
    def _fill_tensor(self, tensor, channel, idx):
        row = idx // self.cards_each_suit
        column = idx % self.cards_each_suit
        tensor[channel, row, column] = 1

    def _is_edge(self, channel):
        return (channel + 1) % self.max_sequential_actions == 0

    def _limit_actions(self, legal_actions):
        legals = legal_actions.copy()
        return [action for action in legals if action in self.limited_actions]

    def _detect_channel_round(self, rounds, player, round_counter):

        # Qui invece sono dopo il round iniziale
        # Innanzi tutto conto quanti rounds ci sono, in questo modo posso capire già il canale di partenza
        start_channel = (round_counter)*self.max_sequential_actions

        # Capito il canale conto le azioni svolte dal giocatore
        # Innanzi tutto controllo se c'è corrispondenza tra round_counter e rounds, ovvero se sto iniziando un nuovo round oppure sto finendo un round già iniziato
        # Ad esempio: round_counter=1 rounds=[[(1, 'call'), (0, 'raise')]] qui sto iniziando un nuovo round e non ho corrispondenza con rounds-1
        current_round = []
        if(round_counter < len(rounds)):
            current_round = rounds[round_counter]

        end_channel = start_channel
        # Ora dovrei aggiungere a start_channel le azioni svolte dal giocatore
        end_channel += np.sum(1 for action in current_round if action[0] == player)

        # if end_channel >= self.max_round*self.max_sequential_actions:
        #     print("end_channel Maggiore di dimensione!!!", end_channel, self.max_round*self.max_sequential_actions)
        #     print(rounds)
        #     for player_index in range(self.num_players):
        #         print(player_index, self.action_tensor[player_index])
        
        return end_channel

    def _fill_rounds(self, tensor, rounds, idx_player, isAction = False):
        # isAction sto trattando azioni o legals

        # print('_fill_rounds FUN', rounds)
        # Scorro ogni singolo round mantenendo l'indice 0, 1 etc, lo uso come riferimento per start_channel
        for round_counter, round_series in enumerate(rounds):
            # print(f"Round {round_counter + 1}:")
            # A questo punto divido le azioni per giocatori
            player_series = {}
            for player_id, serie in round_series:
                if player_id not in player_series:
                    player_series[player_id] = []
                player_series[player_id].append((player_id, serie))

            # print(player_series)
            # Ora scorro player_series e aggiungo le azioni svolte man mano
            for player_id, series in player_series.items():
                start_channel = round_counter*self.max_sequential_actions

                for serie_index, taken in enumerate(series):
                    for player_index in range(self.num_players):
                        if(isAction):
                            tensor[start_channel+serie_index, player_id, :] = np.array([1 if action == taken[1] else 0 for action in self.actions])
                        elif idx_player == player_id:
                            # Sto salvando legals che vanno convertiti
                            # Però devo premurarmi di riapplicare il limited in caso di edge
                            legals = taken[1]
                            if self._is_edge(start_channel+serie_index):
                                # Tengo solo le azioni disponibili in limited_actions
                                legals = self._limit_actions(legals)

                            tensor[start_channel+serie_index, -1, :] = np.array([1 if action in legals else 0 for action in self.actions])
            # print(tensor)
        # print('---------')
                            

    def _fill_sum(self, tensor):

        sum = np.zeros(self.action_tensor_shape[2])
        # Scorro ogni canale
        for channel_index, channel_data in enumerate(tensor):
            # print("Indice canale: ",channel_index, tensor[channel_index, -1, :], "sum",np.sum(channel_data[-1,:]), "all",np.all(channel_data[-1,:] == 0))
            if(channel_index % self.max_sequential_actions == 0):
                sum = channel_data[-2, :].copy()
            
            # Se è tutto 0 allora continuo
            if(np.all(channel_data == 0)):
                continue
            
            for player_index in range(self.num_players):
                sum = np.logical_or(sum, channel_data[player_index, :])
            # print(sum)
            channel_data[-2, :] = sum.copy()
        
    # Funzione per applicare il padding
    def pad_tensor(self, tensor, target_rows, target_cols):
        # Crea una copia del tensore originale
        tensor_copy = tensor.copy()
        
        # Calcola quanto padding è necessario per righe e colonne
        padding_rows = target_rows - tensor_copy.shape[1]
        padding_cols = target_cols - tensor_copy.shape[2]
        
        # Applica il padding alla copia
        tensor_padded = np.pad(tensor_copy, ((0, 0), (0, padding_rows), (0, padding_cols)), 'constant', constant_values=0)
        return tensor_padded

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        # print(state, "round_counter", round_counter, "state['round_counter']",state['round_counter'])
        extracted_state = {}

        public_card = state['public_card']
        hand = state['hand']
        round_counter = 0
        is_over = False

        card_tensor = np.zeros(self.card_tensor_shape, dtype=np.int8)
        action_tensor = np.zeros(self.action_tensor_shape, dtype=np.int8)

        if(self.game.is_over()):
            is_over = True
        
        if(state['round_counter'] > round_counter):
            round_counter = state['round_counter']
        elif(len(self.action_recorder_round) > 1 and state['round_counter'] == 0):
            # Questa condizione evita di aggiungere legal_actions al turno successivo se la partita è finita (es. opp ha foldato)
            # Quindi viene salvato che l'opp ha foldato come actions ma non le legal_actions del turno successivo che non si terrà mai
            is_over = True
            # Questa condizione è importante per i round successivi allo 0 in cui poi finisce il gioco, mi serve per _detect_channel_round
            
        # print("Test recorder legals: ",self.action_recorder_legals, state['legal_actions'], " acted: ",self.action_recorder_round )
        
        # Ri-setto ogni volta le azioni già svolte da entrambi i giocatori su entrambi i tensori, lo so è inefficiente ma evita un problema che mi segno
        # r0 [(0, 'raise')] p1
        # r0 [(0, 'raise')] p1
        # r0 [(0, 'raise')] p1
        # r0 [(0, 'raise'), (1, 'raise')] p0
        # Qui p0 vede di aver fatto il raise solo alla fine, quindi è possibile che lo perda vedendo direttamente il round successivo
        # Per evitarlo potrei ri-settare solo una porzione finale-iniziale degli ultimi 2 action_recorder_round, ma sticazzi
        self._fill_rounds(action_tensor, self.action_recorder_round, state['current_player'], True)
        self._fill_rounds(action_tensor, self.action_recorder_legals, state['current_player'])

        # print(action_tensor)
        
        # obs = np.zeros(36, dtype=np.int8)
        # obs[self.card2index[hand]] = 1
        # if public_card:
        #     obs[self.card2index[public_card]+3] = 1
        # obs[state['my_chips']+6] = 1
        # obs[state['all_chips'][1]+20] = 1
        # extracted_state['obs'] = obs
        
        """
        card_tensor = self.card_tensor[state['current_player']]
        card_channels = card_tensor.shape[-1]

        # Qui gestisco l'inserimento nei vari round, cambia solo la carta da aggiungere
        # if(round_counter == 0):
        #     self._fill_tensor(card_tensor, round_counter, self.card2index[hand])
        # elif round_counter > 0:
        #     self._fill_tensor(card_tensor, round_counter, self.card2index[public_card])
        #     self._fill_tensor(card_tensor, card_channels-1, self.card2index[public_card])

        # if public_card:
        #     self._fill_tensor(card_tensor, card_channels-2, self.card2index[public_card])
        #     self._fill_tensor(card_tensor, card_channels-1, self.card2index[public_card])

        # # print(card_tensor, num_channels)
        action_tensor = self.action_tensor[state['current_player']]

        # if(round_counter != state['round_counter']):
        #     print("OCCHIO DISCREPANZA round_counter: ",round_counter, state['round_counter'], self.action_recorder, self.action_recorder, self.action_recorder_round, self.action_recorder_round)
        #     print('TERMINATION', state)
        #     # state['legal_actions'] = []

        """

        # Aggiungo le legal_actions solo se sono piene e non è finita
        if len(state['legal_actions']) and not is_over:
            channel_legals = self._detect_channel_round(self.action_recorder_round, state['current_player'], round_counter)
            
            # Può capitare che ci siano ultimi stati finali (tipo ricompense o reset) in cui è maggiore, basta ignorarli
            if channel_legals < self.max_round*self.max_sequential_actions:
                # Se mi trovo al limite del channel, vorrei limitare le azioni
                if self._is_edge(channel_legals):
                    # Tengo solo le azioni disponibili in limited_actions
                    state['legal_actions'] = self._limit_actions(state['legal_actions'])

                # Ulteriore controllo per gioco "finito", ho notato che apparentemente azzera round_counter e action_recorder_round
                # Ha le lega_actions vuote o già settate?
                if(np.all(action_tensor[channel_legals, -1, :] == 0)):
                    # print("Set, player: ", state['current_player'],"old legals: ",action_tensor[channel_legals, -1, :], "new legals: ",np.array([1 if action in state['legal_actions'] else 0 for action in self.actions]) )
                    # Modifico esplicitamente il tensore per aggiungere le azioni
                    action_tensor[channel_legals, -1, :] = np.array([1 if action in state['legal_actions'] else 0 for action in self.actions])
                else:
                    if(round_counter == 0 and len(self.action_recorder_round) == 0):
                        # Se il giocatore non ha ancora agito magari è il round iniziale/reset
                        if(not np.all(action_tensor[channel_legals, state['current_player'], :] == 0)):
                            print('----------- ERRORE qui vorrei indagare ', "round_counter: ", round_counter,  "action_recorder_round: ",self.action_recorder_round,"channel_legals: ", channel_legals,"current_player: ", state['current_player'], "current_player: ", action_tensor[channel_legals], "new_legals: ",np.array([1 if action in state['legal_actions'] else 0 for action in self.actions]), "old_legals: ",action_tensor[channel_legals, -1, :], "action_tensor: ", action_tensor)
                            print(state)
                    else:
                        # Posso controllare se il giocatore ha già agito in quel round?
                        if(not np.all(action_tensor[channel_legals, state['current_player'], :] == 0)):
                            print('----------- ERRORE player già agito, strano? ', "self.round_counter: ", round_counter, "round_counter: ", state['round_counter'],  "action_recorder_round: ",self.action_recorder_round,"channel_legals: ", channel_legals,"current_player: ", state['current_player'], "current_player: ", action_tensor[channel_legals], "new_legals: ",[1 if action in state['legal_actions'] else 0 for action in self.actions], "action_tensor: ", action_tensor)
                        elif(not np.all(action_tensor[channel_legals, -1, :] == np.array([1 if action in state['legal_actions'] else 0 for action in self.actions]))):
                            print('----------- ERRORE già setto ma azioni diverse ', "self.round_counter: ", round_counter, "round_counter: ", state['round_counter'],  "action_recorder_round: ",self.action_recorder_round,"channel_legals: ", channel_legals,"current_player: ", state['current_player'], "current_player: ", action_tensor[channel_legals], "new_legals: ",np.array([1 if action in state['legal_actions'] else 0 for action in self.actions]), "old_legals: ",action_tensor[channel_legals, -1, :], "action_tensor: ", action_tensor)
                        elif(np.all(action_tensor[channel_legals, -1, :] == np.array([1 if action in state['legal_actions'] else 0 for action in self.actions]))):
                            1+1
                        else:
                            print('Setto ', "round_counter: ", round_counter,  "action_recorder_round: ",self.action_recorder_round,"channel_legals: ", channel_legals, "actions_attuali nel tensore: ", action_tensor[channel_legals, -1, :], "nuove ",[1 if action in state['legal_actions'] else 0 for action in self.actions])
            else:
                print("channel_legals (",channel_legals,") Maggiore o uguale a ",self.max_round*self.max_sequential_actions, state)
        
        # Qui implemento sum, essenziamente segna come 1 quando un player fa un azione e se lo trascina fino alla fine del sub-round
        self._fill_sum(action_tensor)

        legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions

        # Applico padding
        card_tensor_pad = self.pad_tensor(card_tensor, self.max_rows, self.max_cols)
        action_tensor_pad = self.pad_tensor(action_tensor, self.max_rows, self.max_cols)

        # Fondo i tensori
        obs = np.concatenate((card_tensor_pad, action_tensor_pad), axis=0)

        extracted_state['obs'] = obs

        extracted_state['card_tensor'] = card_tensor
        extracted_state['action_tensor'] = action_tensor

        # print(extracted_state['action_tensor'])
        
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder
        extracted_state['action_recorder_round'] = self.action_recorder_round
        extracted_state['action_recorder_legals'] = self.action_recorder_legals

        # print(state['current_player'], action_tensor, self.action_recorder_round)
        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_card'] = self.game.public_card.get_index() if self.game.public_card else None
        state['hand_cards'] = [self.game.players[i].hand.get_index() for i in range(self.num_players)]
        state['current_round'] = self.game.round_counter
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state
