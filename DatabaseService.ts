import { open } from "@op-engineering/op-sqlite";
import * as Crypto from "expo-crypto";

export class DatabaseService {
  private db: any;
  private readonly vectorDimension = 512;

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

      // Create vector index
      await this.db.execute(`
        CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings(
          libsql_vector_idx(embedding, 'compress_neighbors=float8', 'max_neighbors=50')
        );
      `);
      console.log("Created vector index successfully");

      // Test the table with a simple vector
      const testVector = new Float32Array(this.vectorDimension).fill(0.1);
      await this.db.execute(
        "INSERT INTO embeddings(uuid, content, embedding) VALUES (?, ?, vector(?))",
        [await this.generateUUID(), "test", this.toVector(testVector)]
      );
      console.log("Successfully inserted test vector");
    } catch (error) {
      console.error("Error in open():", error);
      throw error;
    }
  }

  private async generateUUID(): Promise<string> {
    return await Crypto.randomUUID();
  }

  private toVector(embedding: Float32Array): string {
    return `[${Array.from(embedding).join(", ")}]`;
  }

  async storeEmbedding(
    content: string,
    embedding: Float32Array,
    metadata?: any
  ) {
    try {
      console.log("Storing embedding:", {
        content,
        embeddingLength: embedding.length,
      });

      const uuid = await this.generateUUID();
      const result = await this.db.execute(
        "INSERT INTO embeddings(uuid, content, embedding) VALUES (?, ?, vector(?))",
        [uuid, content, this.toVector(embedding)]
      );
      return uuid;
    } catch (error) {
      console.error("Error in storeEmbedding:", error);
      throw error;
    }
  }

  async searchSimilar(queryEmbedding: Float32Array, limit: number = 10) {
    try {
      console.log("Searching with embedding length:", queryEmbedding.length);

      const querySql = `
        WITH vector_matches AS (
          SELECT e.uuid, e.content, vector_distance_cos(e.embedding, vector(?)) as distance
          FROM vector_top_k('idx_embeddings_vector', vector(?), ?) vt
          JOIN embeddings e ON e.rowid = vt.id
        )
        SELECT 
          uuid,
          content,
          distance,
          CASE 
            WHEN distance <= 0.1 THEN 'Very High'
            WHEN distance <= 0.3 THEN 'High'
            WHEN distance <= 0.5 THEN 'Medium'
            WHEN distance <= 0.7 THEN 'Low'
            ELSE 'Very Low'
          END as match_quality
        FROM vector_matches
        ORDER BY distance ASC;
      `;

      const vectorStr = this.toVector(queryEmbedding);
      const results = await this.db.execute(querySql, [
        vectorStr,
        vectorStr,
        limit,
      ]);

      return results.rows.map((row: any) => ({
        content: row.content,
        similarity: 1 - row.distance,
        id: row.uuid,
        matchQuality: row.match_quality,
      }));
    } catch (error) {
      console.error("Error in searchSimilar:", error);
      throw error;
    }
  }

  async close() {
    if (this.db) {
      await this.db.close();
    }
  }

  async populateTestData() {
    const testData = [
      "The quick brown fox jumps over the lazy dog",
      "Machine learning is a subset of artificial intelligence that focuses on data and algorithms",
      "A beautiful sunset painted the sky in shades of orange and purple",
      "The recipe calls for fresh basil, garlic, and extra virgin olive oil",
      "Scientists discovered a new species of deep-sea creature near hydrothermal vents",
      "The ancient ruins revealed secrets about a long-lost civilization",
      "Electric vehicles are becoming increasingly popular as technology improves",
      "The jazz musician improvised a mesmerizing solo on his saxophone",
      "Climate change is affecting weather patterns around the globe",
      "The art exhibition featured works from emerging local artists",
      "Quantum computers could revolutionize cryptography and drug discovery",
      "The chef's signature dish combines traditional and modern cooking techniques",
      "Space exploration has led to numerous technological advancements",
      "The novel tells a compelling story about friendship and redemption",
      "Regular exercise and proper nutrition are essential for good health",
    ];

    console.log("Starting to populate test data...");

    for (const text of testData) {
      try {
        const embedding = new Float32Array(this.vectorDimension);
        // Create a deterministic but varied embedding for testing
        for (let i = 0; i < this.vectorDimension; i++) {
          embedding[i] = Math.sin(i * text.length) * 0.5;
        }

        await this.storeEmbedding(text, embedding);
        console.log(`Stored: "${text.slice(0, 30)}..."`);
      } catch (error) {
        console.error(`Error storing test data: ${text}`, error);
      }
    }

    console.log("Finished populating test data");
  }

  // Helper method to test similarity search
  async testSimilaritySearch(queryText: string) {
    console.log(`\nTesting similarity search for: "${queryText}"`);

    // Create a test embedding for the query
    const queryEmbedding = new Float32Array(this.vectorDimension);
    for (let i = 0; i < this.vectorDimension; i++) {
      queryEmbedding[i] = Math.sin(i * queryText.length) * 0.5;
    }

    const results = await this.searchSimilar(queryEmbedding, 5);

    console.log("\nSearch Results:");
    results.forEach((result, index) => {
      console.log(`\n${index + 1}. Content: "${result.content}"`);
      console.log(`   Similarity: ${(result.similarity * 100).toFixed(2)}%`);
      console.log(`   Match Quality: ${result.matchQuality}`);
    });

    return results;
  }
}
