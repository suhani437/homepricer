import { pgTable, text, serial, integer, real, boolean } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const properties = pgTable("properties", {
  id: serial("id").primaryKey(),
  squareFootage: integer("square_footage").notNull(),
  yearBuilt: integer("year_built").notNull(),
  bedrooms: integer("bedrooms").notNull(),
  bathrooms: real("bathrooms").notNull(),
  garage: integer("garage").notNull().default(0),
  propertyType: text("property_type").notNull(),
  neighborhood: text("neighborhood").notNull(),
  hasPool: boolean("has_pool").notNull().default(false),
  hasFireplace: boolean("has_fireplace").notNull().default(false),
  hasHardwoodFloors: boolean("has_hardwood_floors").notNull().default(false),
  recentlyUpdated: boolean("recently_updated").notNull().default(false),
});

export const predictions = pgTable("predictions", {
  id: serial("id").primaryKey(),
  propertyId: integer("property_id").references(() => properties.id),
  estimatedPrice: real("estimated_price").notNull(),
  confidence: real("confidence").notNull(),
  lowerBound: real("lower_bound").notNull(),
  upperBound: real("upper_bound").notNull(),
});

export const insertPropertySchema = createInsertSchema(properties).omit({
  id: true,
});

export const insertPredictionSchema = createInsertSchema(predictions).omit({
  id: true,
});

export type Property = typeof properties.$inferSelect;
export type InsertProperty = z.infer<typeof insertPropertySchema>;
export type Prediction = typeof predictions.$inferSelect;
export type InsertPrediction = z.infer<typeof insertPredictionSchema>;

export interface PredictionRequest {
  squareFootage: number;
  yearBuilt: number;
  bedrooms: number;
  bathrooms: number;
  garage: number;
  propertyType: string;
  neighborhood: string;
  hasPool: boolean;
  hasFireplace: boolean;
  hasHardwoodFloors: boolean;
  recentlyUpdated: boolean;
}

export interface PredictionResponse {
  estimatedPrice: number;
  confidence: number;
  lowerBound: number;
  upperBound: number;
  featureImportance: Array<{
    feature: string;
    importance: number;
  }>;
}

export interface ModelMetrics {
  rSquared: number;
  rmse: number;
  mae: number;
  lastTrained: string;
}
