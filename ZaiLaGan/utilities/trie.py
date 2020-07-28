from typing import *

class TrieNode():
	def __init__(self, token: str):
		self.token = token
		self.children = {}
		self.isWord = False
		self.frequency = -1

	# Add a child node (connect to next token)
	def addChild(self, child):
		self.children[child.token] = child

	# Record that the node is end of a word
	def setIsWord(self):
		self.isWord = True

	# Record word frequency
	def setWordFrequency(self, frequency: int):
		self.frequency = frequency

class Trie():
	def __init__(self):
		self.root = TrieNode(None)

	# Add a new word into trie
	def addWord(self, word: str, frequency: int):
		currentNode = self.root
		for i in range(len(word)):
			token = word[i]
			# Check if the token has already existed in the trie
			if(token in currentNode.children):
				currentNode = currentNode.children[token]
			else:
				nextNode = TrieNode(token)
				currentNode.children[token] = nextNode
				currentNode = nextNode
			# Check if this is the last token
			if(i == len(word)-1):
				currentNode.setIsWord()
				currentNode.setWordFrequency(frequency)

	# Get word frequency
	def getWordFreq(self, word: str) -> int:
		node = self.root
		for token in word:
			if token not in node.children:
				return -1
			node = node.children[token]
		return int(node.frequency)