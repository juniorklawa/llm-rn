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
import { DatabaseService } from "./DatabaseService";

export default function App() {
  const [isReady, setIsReady] = useState(false);
  const [model, setModel] = useState<any>(null);
  const [db, setDb] = useState<DatabaseService | null>(null);
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
      const database = new DatabaseService(loadedModel);
      await database.open();

      // Add test data
      setStatus("Adding test data...");
      await database.populateTestData();

      // Run demonstration
      setStatus("Running similarity search demonstration...");
      await database.demonstrateExpectedMatches();

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
      setStatus("Storing text...");
      const id = await db.storeEmbedding(inputText);
      setStatus(`Stored with ID: ${id}`);
      setInputText("");
    } catch (error) {
      setStatus(`Error storing text: ${error.message}`);
      console.error("Store error:", error);
    }
  };

  const searchTexts = async () => {
    if (!searchText.trim() || !db) return;

    try {
      setStatus("Searching...");
      const searchResults = await db.searchSimilar(searchText, 5);
      setResults(searchResults);
      setStatus("Search complete");
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
          onPress={searchTexts}
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
              Similarity: {result.similarity.toFixed(1)}% ({result.matchQuality}
              )
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
