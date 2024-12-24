import discord
import asyncio
import RT_b2s
import data_pipeline

class MyClient(discord.Client):
    
    async def on_ready(self):
        print('Logged on as', self.user)


    async def check_dst(self):
        while True:
            await asyncio.sleep(30)
            print("checked")

    async def on_message(self, message):
        global messages
        # don't respond to ourselves

        if message.author == self.user:
            return
        
        if isinstance(message.channel, discord.channel.DMChannel):
            if(message.content =="Marvin enhance"):
                await message.channel.send("Coming right up")
                data_pipeline.run_all()
                RT_b2s.enhance_latest()
                await message.channel.send(file=discord.File('latest.png'))
              
        else:
           print("I cannot do that sorry")
                       

if __name__ == '__main__':	
    messages = []
    intents = discord.Intents.default()
    intents.message_content  = True
    intents.messages         = True
    intents.reactions        = True

    bot = MyClient(intents=intents)
   
    with open('discordcreds.txt', 'r') as file:
        cred = file.read().rstrip()
    bot.run(cred)