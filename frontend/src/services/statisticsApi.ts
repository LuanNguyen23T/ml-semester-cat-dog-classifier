import axios from 'axios';

import type { StatisticsFilters, StatisticsResponse } from '../types/statistics';


const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
});


export const statisticsApi = {
  getStatistics: async (filters: StatisticsFilters): Promise<StatisticsResponse> => {
    const { data } = await api.get<StatisticsResponse>('/statistics', { params: filters });
    return data;
  },
};
