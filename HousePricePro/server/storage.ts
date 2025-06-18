import { properties, predictions, type Property, type InsertProperty, type Prediction, type InsertPrediction } from "@shared/schema";

export interface IStorage {
  getProperty(id: number): Promise<Property | undefined>;
  createProperty(property: InsertProperty): Promise<Property>;
  createPrediction(prediction: InsertPrediction): Promise<Prediction>;
  getPrediction(id: number): Promise<Prediction | undefined>;
}

export class MemStorage implements IStorage {
  private properties: Map<number, Property>;
  private predictions: Map<number, Prediction>;
  private currentPropertyId: number;
  private currentPredictionId: number;

  constructor() {
    this.properties = new Map();
    this.predictions = new Map();
    this.currentPropertyId = 1;
    this.currentPredictionId = 1;
  }

  async getProperty(id: number): Promise<Property | undefined> {
    return this.properties.get(id);
  }

  async createProperty(insertProperty: InsertProperty): Promise<Property> {
    const id = this.currentPropertyId++;
    const property: Property = { 
      ...insertProperty, 
      id,
      garage: insertProperty.garage ?? 0,
      hasPool: insertProperty.hasPool ?? false,
      hasFireplace: insertProperty.hasFireplace ?? false,
      hasHardwoodFloors: insertProperty.hasHardwoodFloors ?? false,
      recentlyUpdated: insertProperty.recentlyUpdated ?? false,
    };
    this.properties.set(id, property);
    return property;
  }

  async createPrediction(insertPrediction: InsertPrediction): Promise<Prediction> {
    const id = this.currentPredictionId++;
    const prediction: Prediction = { 
      ...insertPrediction, 
      id,
      propertyId: insertPrediction.propertyId ?? null
    };
    this.predictions.set(id, prediction);
    return prediction;
  }

  async getPrediction(id: number): Promise<Prediction | undefined> {
    return this.predictions.get(id);
  }
}

export const storage = new MemStorage();
