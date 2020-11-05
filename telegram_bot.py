# from knockknock import telegram_sender
import requests


def telegram_bot_sendtext(bot_message):
    # Bot Name: ml_contest_bot
    # Group Name: Machine_learning_contest
    # bot_token = '862315976:AAE1Vzu9L5J1liCNrrCVnWdNTS4vGKvRZww'
    # bot_chatID = '-324127062'

    bot_message_str = str(bot_message)
    bot_token = '862315976:AAE1Vzu9L5J1liCNrrCVnWdNTS4vGKvRZww'
    bot_chatID = '-324127062'
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message_str

    response = requests.get(send_text)
    print("Updated to telegram via bot")
    return response.json()


test = telegram_bot_sendtext("Staring the notebook for Machine_learning_contest")
