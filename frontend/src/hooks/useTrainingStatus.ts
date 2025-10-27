import { useQuery } from "@tanstack/react-query";
import { fetchStatus, TrainingStatus } from "../lib/api";

export const useTrainingStatus = () =>
  useQuery<TrainingStatus, Error>({
    queryKey: ["training-status"],
    queryFn: fetchStatus,
    refetchInterval: 5000
  });
