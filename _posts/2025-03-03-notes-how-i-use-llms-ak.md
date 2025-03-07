---
layout: post
title: Notes on Ho I use LLMs from Andrej Karpathy
---

Reference [YouTube video](https://youtu.be/EWvNQjAaOHw).

### General Tipps

- Keep chats as short as you can. For switching topic, start a new chat, this wipes the context window of tokens and reset them back to cero.
  - Having long conversations i.e. a lot of tokens in the context window, the model could find that distracting and therefore the accuracy performance of the model could decrease.
  - The more tokens in the context window, the more expensive it is (by a little bit) to sample the next tokens in a sequence.

### Using Tools

- Assistants use tools like web-search, calculator, code, etc. and put it in the context window.
- **Web-Search**: 
  - Perplexity.AI great for searching the web, powerful for getting updated info, which were not included during the pre-training stage.
  - Most models offer web search under the hood.
- **Deep-Research**:
  - Combination of web search and thinking.
  - "Almost like a custom research paper" on any topic. Big output tokens.
  - It rolled out a query for a long time - model will spend tens of minutes to come to an answer.
    - Search on many web pages, many papers, etc.
 - Perplexity.AI offers also deep research.

### Writing Code

Available tools:  
- ChatGPT: Advanced Data Analysis
- Claude: Artifacts
- [Cursor](https://www.cursor.com): Composer -> Best now
- VSCode
- [WindSurf](https://windsurfai.org/de)

#### Cursor

- Files in the file system are accessible to the model. 
- Using the Composer "Cmd + I" feature:
    - [Vibe Coding](https://x.com/karpathy/status/1886192184808149383?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1886192184808149383%7Ctwgr%5E490b0284a84e4ff5fe331cdbc62dc7ef934899ad%7Ctwcon%5Es1_c10&ref_url=https%3A%2F%2Fresearch.aimultiple.com%2Fvibe-coding%2F), gives commands and let the LLM to code for you. Andrej used it with Claude Sonnet.
    - "Setup a new React 15 starter project".
    - "Delete the boilerplate and just setup a bare tic tac toe grid in the middle of the page".
    - "Clicking on any cell should create an X or an O inside it, alternating".
- Has built-in commands like:
    - "Cmd + K" to change a line of code.
    - "Cmd + L" explain a line of code.
    - "Cmd + I" opens composer feature.

### Audio

- **How does it work?**:
  - Chunk audio into tokens.
  - Break down audio into spectrogram to see all different frequencies.
  - Snap little windows and quantize them into tokens.
  - E.g. at the end its possible to have a vocabulary of 100,000 possible little audio chunks.
  - Train the model with those audio chunks, so that the model can understand those little pieces of audio.
- [Notebooklm](https://notebooklm.google.com/?pli=1) from Google create a podcast from text/file.
- [Podcast Histories of Mysteries](https://open.spotify.com/show/3K4LRyMCP44kBbiOziwJjb?si=8f6b81f7c7864f11&nd=1&dlsi=9c652091e8384690) - An AI podcast made by Andrej Karpathy. Used [NotebookLM](https://notebooklm.google.com) for the creation. 

### Images

- Goal: **Re-represent images as streams of tokens**.
- **How it works?** From an image:
  - Create a rectangular grid.
  - Chop it up into little patches -> an image is a sequence of patches.
  - Quantize for every patch -> goal is to come with a vocabulary of e.g. 100,000 possible patches.
  - Represent each patch using just the closest patch in the vocabulary.
  - Put the stream of tokens into context windows.
  - Train your models with those tokens.
- [Ideogram.AI](https://ideogram.ai/t/explore) - Generate AI images.
- Output from an image is not fully done in the model. E.g. Dalle 3 is a separate model that takes text and creates images.
- With the prompt: "Generate an image that summarizes today." under the hood:
  - Will create a caption for that image.
  - That caption is sent to a separate model that is an image generator model and returns the image.

> Hint: Interesting that LLMs doesn't know that some of the tokens happen to be text, some to be audio and some images. It just models statistical patterns of tokens streams.   
> Only at the encoding and decoder is where we know how different modalities (audio, text, images) are encode/decode.

### Video 

- Andrej is not sure if for ex. ChatGPT in the mobile app uses video, but instead they used image frames like 1 image per second.
- List of video generated products from [Hearther Cooper on tweeter](https://x.com/HBCoop_/status/1891525719290777696)

### ChatGPT Memory Feature

- Every time a new chat gets started all tokens get wiped clean, but ChatGPT has a functionality to save information from chat to chat. It has to be invoked by saying: Please remember this. "Memory updated" is then shown in the UI.
- The memory bank is like a database of knowledge about the user.
  - This is always prepended to all conversations and the model has always access to it.
- Greatness is overtime by keeping many information in the memory bank. Result: ChatGPT starts to know you better.

 ### For What To Use Assistants

 - **Reading Papers**: Upload the paper, read the paper yourself, ask for clarifications.
 - **Reading Books**: Give the input: "We are reading the Book x, summarize", then add the chapter into the context windows and ask questions.
 - **Python Interpreter**: Integration of the LLM with Python interpreter.
 - **ChatGPT Data Analysis**: Give data, use e.g. "now plot this. Use log scale for y axis".
 - **Claude Artifacts**: Write code to create apps and tools.
    - Given text in the context window, use:
        - "Generate x flashcards from the text"
        - "Now use Artifacts feature to write a flashcard app to test. these" That creates code and an app.
    - "Create conceptual diagram of a book chapter".
    - Examples of more artifacts [Claude Artifacts](https://claudeartifacts.com).

### Assistant Apps Available (2025): 

- OpenAI released ChatGPT in 2022
- Gemini: Google's version
- Meta AI: Meta's version
- Copilot: Microsoft's version
- Claude: Anthropic's version
- Grok: xAI's version
- DeepSeek: DeepSeek (Chinese)
- Le Chat: Mistral's version (French)

## Links

- [Ranking Leaderboard LLMArena](https://lmarena.ai/?leaderboard)
- [Ranking Leaderboard Scale](https://scale.com/leaderboard)
- [Perplexity AI](https://www.perplexity.ai) - Have websearch, deep research buit-in, and Host DeepSeek R1.
- [Claude Artifacts](https://claudeartifacts.com)