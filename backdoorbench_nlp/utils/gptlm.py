'''
This code is highly dependent on the official implementation of ONION: https://github.com/thunlp/ONION
The paths to clean & posion datasets are modified in order to fit the overall structure of Backdoorbench_NLP.
Besides, an .yaml file is added to store the hyperparameters.

MIT License

Copyright (c) 2021 THUNLP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import math
import torch
import numpy as np
class GPT2LM:
    def __init__(self, use_tf=False, device=None, little=False):
        """
        :param bool use_tf: If true, uses tensorflow GPT-2 model.
        :Package Requirements:
            * **torch** (if use_tf = False)
            * **tensorflow** >= 2.0.0 (if use_tf = True)
            * **transformers**

        Language Models are Unsupervised Multitask Learners.
        `[pdf] <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`__
        `[code] <https://github.com/openai/gpt-2>`__
        """
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import transformers
        self.use_tf = use_tf
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")

        if use_tf:
            self.lm = transformers.TFGPT2LMHeadModel.from_pretrained("gpt2")
        else:
            self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", from_tf=False)
            self.lm.to(device)

        
    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        if self.use_tf:
            import tensorflow as tf
            ipt = self.tokenizer(sent, return_tensors="tf", verbose=False)
            ret = self.lm(ipt)[0]
            loss = 0
            for i in range(ret.shape[0]):
                it = ret[i]
                it = it - tf.reduce_max(it, axis=1)[:, tf.newaxis]
                it = it - tf.math.log(tf.reduce_sum(tf.exp(it), axis=1))[:, tf.newaxis]
                it = tf.gather_nd(it, list(zip(range(it.shape[0] - 1), ipt.input_ids[i].numpy().tolist()[1:])))
                loss += tf.reduce_mean(it)
                break
            return math.exp(-loss)
        else:
            ipt = self.tokenizer(sent, return_tensors="pt", verbose=False,  )
            # print(ipt)
            # print(ipt.input_ids)
            try:
                ppl = math.exp(self.lm(input_ids=ipt['input_ids'].cuda(),
                                 attention_mask=ipt['attention_mask'].cuda(),
                                 labels=ipt.input_ids.cuda())[0])
            except RuntimeError:
                ppl = np.nan
            return ppl





