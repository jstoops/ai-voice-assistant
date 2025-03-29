# AI Voice Assistant

<img src="/images/screen.webp" alt="AI Voice Assistant" />

## Features

- Allows you to ask questions and get witty, concise answers about what you're working on
- Uses webcam to see you, objects held and room around you
- Uses desktop image feed to see what you're working on
- Uses multiple AI models to convert text-to-speech, speech-to-text, and answer questions using the images streamed from the webcam and desktop
- Uses OpenAIs TTS-1, TTS-1-HD or GPT-4o-mini-TTS model to convert text to natural sounding spoken text
- Uses OpenAIs GTP-4o or Google's Gemini 1.5 Flash model for turning user questions & images into useful help & banter
- Change the AI Assistant to be more formal, less witty, ask clarifying questions or provide more detailed answers by adjusting the prompt
- Easy to configure to use [different voices](https://platform.openai.com/docs/guides/text-to-speech) for the AI Assistant to respond with various accents, emotional ranges, intonation, impressions, speed of speech, tone, etc.
- For example, download this code, run it and ask it to [start improving it's own code](https://tvtropes.org/pmwiki/pmwiki.php/Main/GrewBeyondTheirProgramming)

AI Voice Assistant uses the following technologies:

- [CV](https://github.com/opencv/opencv-python): A library that includes several hundreds of computer vision algorithms. This project uses VideoCapture class for video capturing from video files, image sequences or cameras and the imencode function that compresses the image and stores it in the memory buffer that is resized to fit the result.
- [OpenAI Python API library](https://github.com/openai/openai-python): Provides convenient access to the OpenAI REST API from any Python 3.8+ application. The library includes type definitions for all request params and response fields, and offers both synchronous and asynchronous clients powered by [httpx](https://github.com/encode/httpx).
- [PyTorch](https://pytorch.org/): A python package that provides tensor computation, like [NumPy](https://numpy.org/), with strong GPU acceleration and deep neural networks built on a tape-based autograd system.
- [TorchAudio](https://github.com/pytorch/audio): Apply PyTorch to the audio domain by providing strong GPU acceleration, having a focus on trainable features through the autograd system, and having consistent style (tensor names and dimension names). Therefore, it is primarily a machine learning library and not a general signal processing library. The benefits of PyTorch can be seen in torchaudio through having all the computations be through PyTorch operations which makes it easy to use and feel like a natural extension.
- [TorchVision](https://github.com/pytorch/vision): Consists of popular datasets, model architectures, and common image transformations for computer vision.
- [Whisper](https://github.com/openai/whisper): A general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.
- [SoundFile](https://github.com/bastibe/python-soundfile): An audio library based on [libsndfile](http://www.mega-nerd.com/libsndfile/), [CFFI](https://cffi.readthedocs.io/) and [NumPy](https://numpy.org/).
- [Base64](https://github.com/mayeut/pybase64): Used to encode images sent from webcam & desktop to base64.
- [Lock](https://github.com/wdbm/lock): Implements advisory file locking that enables concurrent processes to interact with the same file without conflict, provided they first check for the existence of a lock held by a different process.
- [Thread](https://thread.ngjx.org/docs): Provides out-of-the-box solution with multi-threaded processing and fetching values from a completed thread, etc.
- [DotEnv](): reads key-value pairs from a .env file and can set them as environment variables. It helps in the development of applications following the [12-factor principles](https://12factor.net/).
- [LangChain](https://python.langchain.com/): This library aims to assist in the development of truly powerful apps that comes when you can combine them with other sources of computation or knowledge.
- [LangChain Community](https://python.langchain.com/api_reference/community/index.html): Contains third-party integrations that implement the base interfaces defined in LangChain Core, making them ready-to-use in any LangChain application.
- [LangChain Core](https://python.langchain.com/api_reference/core/index.html): Contains the base abstractions that power the rest of the LangChain ecosystem. These abstractions are designed to be as modular and simple as possible. Examples of these abstractions include those for language models, document loaders, embedding models, vectorstores, retrievers, and more.
- [LangChain OpenAI](https://python.langchain.com/docs/integrations/components/): This package contains the LangChain integrations for OpenAI through their openai SDK.
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/): Provides Python bindings for PortAudio v19, the cross-platform audio I/O library. With PyAudio, you can easily use Python to play and record audio on a variety of platforms, such as GNU/Linux, Microsoft Windows, and Apple macOS.
- [SpeechRecognition](https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst): Library for performing speech recognition, with support for several engines and APIs, online and offline.
- [FFmpeg](https://python-ffmpeg.readthedocs.io/en/stable/): A python binding for [FFmpeg](https://ffmpeg.org/), a complete, cross-platform solution to record, convert and stream audio and video.
- [Numba](https://numba.pydata.org/): A just-in-time compiler for numerical functions in python. It uses the LLVM compiler project to generate machine code from Python syntax. Numba can compile a large subset of numerically-focused Python, including many NumPy functions. Additionally, Numba has support for automatic parallelization of loops, generation of GPU-accelerated code, and creation of ufuncs and C callbacks.
- [More Itertools](https://github.com/more-itertools/more-itertools): A library used to compose elegant solutions for a variety of problems with the functions it provides. A collection of building blocks, recipes, and routines for working with Python iterables.
