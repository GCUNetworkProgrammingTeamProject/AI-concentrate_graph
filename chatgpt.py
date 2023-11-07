import openai
'''
openai.api_key = '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'

messages = []
content = "코끼리가 뭐야?"
messages.append({"role": "user", "content": content})
print(messages)

completion = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    messages = messages
)
com = completion
res = completion.choices[0].message.content
print(com)

messages.append({"role": "assistant", "content": res})
print(messages)
'''
def chatgpt(c):
    openai.api_key = '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'

    messages = []
    # content = "코끼리가 뭐야?"
    content = c
    messages.append({"role": "user", "content": content})
    # print(messages)

    completion = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = messages
    )
    res = completion.choices[0].message.content
    return res
    