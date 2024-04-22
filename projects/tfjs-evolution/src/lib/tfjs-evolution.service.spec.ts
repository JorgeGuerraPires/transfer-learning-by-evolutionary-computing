import { TestBed } from '@angular/core/testing';

import { TfjsEvolutionService } from './tfjs-evolution.service';

describe('TfjsEvolutionService', () => {
  let service: TfjsEvolutionService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(TfjsEvolutionService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
