import { StatusBar } from "expo-status-bar";
import React, { useEffect } from "react";
import { Button, StyleSheet, Text, View } from "react-native";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import * as use from "@tensorflow-models/universal-sentence-encoder";

export default function App() {
  useEffect(() => {
    tf.ready().then(() => {
      console.log("TensorFlow.js is ready");
    });
  }, []);

  const init = async () => {
    // initialize state variables
    // console.log("App is initializing services...")

    // Load the model.
    try {
      use.load().then((model: any) => {
        // Embed an array of sentences.
        const sentences = ["Hello.", "How are you?"];
        model.embed(sentences).then((embeddings: any) => {
          // `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
          // So in this example `embeddings` has the shape [2, 512].
          embeddings.print(true /* verbose */);
        });
      });
    } catch (err) {
      console.log(`ERROR: ${err}`);
    }
  };

  useEffect(() => {
    console.log("App is starting...");
  }, []);

  return (
    <View style={styles.container}>
      <Text>Open up App.js to start working on your app!</Text>
      <StatusBar style="auto" />
      <Button title="Init" onPress={init} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
  },
});
