import { StatusBar } from "expo-status-bar";
import React, { useEffect, useState } from "react";
import {
  Button,
  StyleSheet,
  Text,
  View,
  TextInput,
  ScrollView,
} from "react-native";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import * as use from "@tensorflow-models/universal-sentence-encoder";

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [model, setModel] = useState<any>(null);
  const [inputText, setInputText] = useState("");
  const [searchText, setSearchText] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const [status, setStatus] = useState("Initializing...");

  useEffect(() => {
    initializeServices();
  }, []);

  const initializeServices = async () => {
    try {
      setStatus("Initializing TensorFlow.js...");
      await tf.ready();

      setStatus("Loading Universal Sentence Encoder...");
      const loadedModel = await use.load();
      setModel(loadedModel);

      setStatus("Initializing database...");
      const database = new DatabaseService();
      await database.open();
      setDb(database);

      setStatus("Ready!");
      setIsReady(true);
    } catch (error) {
      setStatus(`Error: ${error.message}`);
      console.error(error);
    }
  };

  const storeText = async () => {
    if (!inputText.trim() || !model || !db) return;

    try {
      setStatus("Generating embedding...");
      const embeddings = await model.embed([inputText]);
      const embedding = await embeddings.array();
      console.log("Generated store embedding:", {
        text: inputText,
        embeddingLength: embedding[0].length,
      });

      setStatus("Storing in database...");
      const id = await db.storeEmbedding(
        inputText,
        new Float32Array(embedding[0]),
        { timestamp: Date.now() }
      );

      setStatus(`Stored with ID: ${id}`);
      setInputText("");
    } catch (error) {
      setStatus(`Error storing text: ${error.message}`);
      console.error("Store error:", error);
    }
  };

  const searchSimilar = async () => {
    if (!searchText.trim() || !model || !db) return;

    try {
      setStatus("Generating search embedding...");
      const embeddings = await model.embed([searchText]);
      const embedding = await embeddings.array();
      console.log("Generated search embedding:", {
        text: searchText,
        embeddingLength: embedding[0].length,
      });

      setStatus("Searching...");
      const searchResults = await db.searchSimilar(
        new Float32Array(embedding[0]),
        5,
        0.1
      );

      setResults(searchResults);
      setStatus(`Found ${searchResults.length} results`);
    } catch (error) {
      setStatus(`Error searching: ${error.message}`);
      console.error("Search error:", error);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar style="auto" />

      <Text style={styles.status}>{status}</Text>

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Enter text to store"
          multiline
          editable={isReady}
        />
        <Button
          title="Store"
          onPress={storeText}
          disabled={!isReady || !inputText.trim()}
        />
      </View>

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={searchText}
          onChangeText={setSearchText}
          placeholder="Enter text to search"
          multiline
          editable={isReady}
        />
        <Button
          title="Search"
          onPress={searchSimilar}
          disabled={!isReady || !searchText.trim()}
        />
      </View>

      <ScrollView style={styles.results}>
        {results.map((result, index) => (
          <View key={result.id} style={styles.resultItem}>
            <Text style={styles.resultText}>
              {index + 1}. {result.content}
            </Text>
            <Text style={styles.similarity}>
              Similarity: {(result.similarity * 100).toFixed(1)}%
            </Text>
          </View>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    padding: 20,
    paddingTop: 60,
  },
  status: {
    textAlign: "center",
    marginBottom: 20,
    color: "#666",
  },
  inputContainer: {
    marginBottom: 20,
  },
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    borderRadius: 5,
    padding: 10,
    marginBottom: 10,
    minHeight: 60,
  },
  results: {
    flex: 1,
  },
  resultItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: "#eee",
  },
  resultText: {
    fontSize: 16,
  },
  similarity: {
    fontSize: 14,
    color: "#666",
    marginTop: 5,
  },
});
