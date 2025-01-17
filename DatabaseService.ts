import { open } from "@op-engineering/op-sqlite";
import * as Crypto from "expo-crypto";
import * as use from "@tensorflow-models/universal-sentence-encoder";

export class DatabaseService {
  private db: any;
  private model: use.UniversalSentenceEncoder | null = null;
  private readonly vectorDimension = 512;

  constructor(model?: use.UniversalSentenceEncoder) {
    this.model = model || null;
  }

  setModel(model: use.UniversalSentenceEncoder) {
    this.model = model;
  }

  private async generateEmbedding(text: string): Promise<Float32Array> {
    if (!this.model) {
      throw new Error("Universal Sentence Encoder model not initialized");
    }
    const embeddings = await this.model.embed([text]);
    const embedding = await embeddings.array();
    return new Float32Array(embedding[0]);
  }

  async open() {
    try {
      this.db = await open({
        name: "vector_db.db",
      });

      console.log("Database opened successfully");

      // Create the vector table with proper BLOB type
      await this.db.execute(`DROP TABLE IF EXISTS embeddings;`);
      console.log("Dropped existing table if any");

      await this.db.execute(`
        CREATE TABLE embeddings (
          uuid TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          embedding F32_BLOB(${this.vectorDimension})
        );
      `);
      console.log("Created table successfully");

      // Create vector index with improved parameters
      await this.db.execute(`
        CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings(
          libsql_vector_idx(embedding, 'compress_neighbors=float8', 'max_neighbors=100')
        );
      `);
      console.log("Created vector index successfully");
    } catch (error) {
      console.error("Error in open():", error);
      throw error;
    }
  }

  private async generateUUID(): Promise<string> {
    return await Crypto.randomUUID();
  }

  private toVector(embedding: Float32Array): string {
    try {
      // Ensure we're working with valid numbers
      const validNumbers = Array.from(embedding).map((n) =>
        Number.isFinite(n) ? n : 0
      );
      return `[${validNumbers.join(",")}]`;
    } catch (error) {
      console.error("Error converting embedding to vector string:", error);
      throw new Error("Failed to convert embedding to vector string");
    }
  }

  async storeEmbedding(content: string, embedding?: Float32Array) {
    try {
      const vectorEmbedding =
        embedding || (await this.generateEmbedding(content));

      console.log("Storing embedding:", {
        content,
        embeddingLength: vectorEmbedding.length,
      });

      const uuid = await this.generateUUID();
      const result = await this.db.execute(
        "INSERT INTO embeddings(uuid, content, embedding) VALUES (?, ?, vector(?))",
        [uuid, content, this.toVector(vectorEmbedding)]
      );
      return uuid;
    } catch (error) {
      console.error("Error in storeEmbedding:", error);
      throw error;
    }
  }

  async searchSimilar(queryText: string, k: number = 10) {
    try {
      const queryEmbedding = await this.generateEmbedding(queryText);
      console.log("Searching with embedding length:", queryEmbedding.length);

      const querySql = `
        SELECT 
          uuid,
          content,
          vector_distance_cos(embedding, vector(?)) as cos_distance
        FROM embeddings
        ORDER BY cos_distance ASC
        LIMIT ?;
      `;

      const vectorStr = this.toVector(queryEmbedding);
      console.log("Vector string length:", vectorStr.length);

      const results = await this.db.execute(querySql, [vectorStr, k]);

      return results.rows.map((row: any) => {
        const distance = row.cos_distance;
        const similarity = Math.round((1 - distance) * 100);
        return {
          content: row.content,
          similarity: similarity,
          id: row.uuid,
          matchQuality: this.getMatchQuality(distance),
          distance: distance,
        };
      });
    } catch (error) {
      console.error("Error in searchSimilar:", error);
      throw error;
    }
  }

  private getMatchQuality(distance: number): string {
    if (distance <= 0.15) return "Excellent";
    if (distance <= 0.3) return "Very Good";
    if (distance <= 0.45) return "Good";
    if (distance <= 0.6) return "Fair";
    return "Poor";
  }

  async populateTestData() {
    if (!this.model) {
      throw new Error(
        "Cannot populate test data without Universal Sentence Encoder model"
      );
    }

    const testData = [
      // Technology and AI
      "Artificial intelligence and machine learning are transforming the technology landscape",
      "Deep learning models have achieved remarkable success in natural language processing",
      "Neural networks can recognize patterns in complex datasets",
      "The future of computing lies in quantum technologies",

      // Nature and Environment
      "Climate change poses significant challenges to global ecosystems",
      "Rainforests are crucial for maintaining Earth's biodiversity",
      "Ocean conservation efforts protect marine life and coral reefs",
      "Sustainable energy solutions are essential for environmental protection",

      // Food and Cooking
      "Traditional Italian cuisine emphasizes fresh, high-quality ingredients",
      "Modern molecular gastronomy combines science and culinary arts",
      "The art of baking requires precision and patience",
      "Farm-to-table restaurants promote local and seasonal ingredients",

      // Arts and Culture
      "Classical music continues to inspire contemporary composers",
      "Abstract expressionism revolutionized modern art in the 20th century",
      "Digital art is gaining recognition in mainstream galleries",
      "Photography captures moments that tell powerful stories",

      // Science and Discovery
      "Space exploration reveals new mysteries about our universe",
      "Genetic research advances our understanding of human health",
      "Neuroscience studies the complexity of the human brain",
      "Paleontology uncovers evidence of prehistoric life",
    ];

    console.log("Starting to populate test data...");

    for (const text of testData) {
      try {
        await this.storeEmbedding(text);
        console.log(`Stored: "${text.slice(0, 30)}..."`);
      } catch (error) {
        console.error(`Error storing test data: ${text}`, error);
      }
    }

    console.log("Finished populating test data");
  }

  async demonstrateExpectedMatches() {
    const testQueries = [
      {
        query: "What's new in artificial intelligence and machine learning?",
        expectedTopics: ["AI", "Deep Learning", "Neural Networks"],
      },
      {
        query: "How does climate change affect biodiversity?",
        expectedTopics: ["Climate Change", "Biodiversity", "Ecosystems"],
      },
      {
        query: "Modern cooking techniques in Italian cuisine",
        expectedTopics: ["Modern Gastronomy", "Italian Cuisine", "Cooking"],
      },
    ];

    console.log("\n=== Demonstration of Expected Matches ===\n");

    for (const test of testQueries) {
      console.log(`\nQuery: "${test.query}"`);
      console.log(`Expected topics: ${test.expectedTopics.join(", ")}`);
      console.log("\nActual results:");

      const results = await this.searchSimilar(test.query, 5);

      console.log("\nSearch Results:");
      results.forEach((result, index) => {
        console.log(`\n${index + 1}. Content: "${result.content}"`);
        console.log(`   Similarity: ${result.similarity.toFixed(2)}%`);
        console.log(`   Match Quality: ${result.matchQuality}`);
        console.log(`   Distance: ${result.distance.toFixed(4)}`);
      });

      console.log("\n-----------------------------------");
    }
  }
}
