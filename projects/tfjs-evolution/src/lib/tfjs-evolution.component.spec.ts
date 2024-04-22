import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TfjsEvolutionComponent } from './tfjs-evolution.component';

describe('TfjsEvolutionComponent', () => {
  let component: TfjsEvolutionComponent;
  let fixture: ComponentFixture<TfjsEvolutionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TfjsEvolutionComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(TfjsEvolutionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
