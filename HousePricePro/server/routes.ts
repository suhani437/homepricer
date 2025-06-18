import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertPropertySchema, type PredictionRequest, type PredictionResponse, type ModelMetrics } from "@shared/schema";
import { spawn } from "child_process";
import path from "path";

export async function registerRoutes(app: Express): Promise<Server> {
  
  // Predict house price endpoint
  app.post("/api/predict", async (req, res) => {
    try {
      const predictionData = req.body as PredictionRequest;
      
      // Validate the input data
      const validatedData = insertPropertySchema.parse(predictionData);
      
      // Call Python ML model
      const prediction = await callPythonModel(predictionData);
      
      // Store the property and prediction
      const property = await storage.createProperty(validatedData);
      await storage.createPrediction({
        propertyId: property.id,
        estimatedPrice: prediction.estimatedPrice,
        confidence: prediction.confidence,
        lowerBound: prediction.lowerBound,
        upperBound: prediction.upperBound,
      });
      
      res.json(prediction);
    } catch (error) {
      console.error("Prediction error:", error);
      res.status(400).json({ 
        message: error instanceof Error ? error.message : "Prediction failed" 
      });
    }
  });

  // Get model metrics endpoint
  app.get("/api/model-metrics", async (req, res) => {
    try {
      const metricsData = await getModelMetrics();
      res.json(metricsData);
    } catch (error) {
      console.error("Model metrics error:", error);
      res.status(500).json({ 
        message: "Failed to retrieve model metrics" 
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

async function callPythonModel(data: PredictionRequest): Promise<PredictionResponse> {
  return new Promise((resolve, reject) => {
    const pythonPath = process.env.PYTHON_PATH || "python3";
    const scriptPath = path.join(process.cwd(), "server", "ml_model.py");
    
    const pythonProcess = spawn(pythonPath, [scriptPath]);
    
    let dataString = "";
    let errorString = "";
    
    pythonProcess.stdout.on("data", (data) => {
      dataString += data.toString();
    });
    
    pythonProcess.stderr.on("data", (data) => {
      errorString += data.toString();
    });
    
    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python process failed: ${errorString}`));
        return;
      }
      
      try {
        const result = JSON.parse(dataString);
        resolve(result);
      } catch (parseError) {
        reject(new Error(`Failed to parse Python output: ${parseError}`));
      }
    });
    
    // Send input data to Python process
    pythonProcess.stdin.write(JSON.stringify(data));
    pythonProcess.stdin.end();
  });
}

async function getModelMetrics(): Promise<ModelMetrics> {
  return new Promise((resolve, reject) => {
    const pythonPath = process.env.PYTHON_PATH || "python3";
    const scriptPath = path.join(process.cwd(), "server", "ml_model.py");
    
    const pythonProcess = spawn(pythonPath, [scriptPath, "--metrics"]);
    
    let dataString = "";
    let errorString = "";
    
    pythonProcess.stdout.on("data", (data) => {
      dataString += data.toString();
    });
    
    pythonProcess.stderr.on("data", (data) => {
      errorString += data.toString();
    });
    
    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python process failed: ${errorString}`));
        return;
      }
      
      try {
        const result = JSON.parse(dataString);
        resolve(result);
      } catch (parseError) {
        reject(new Error(`Failed to parse Python output: ${parseError}`));
      }
    });
  });
}
