import {BaseModel, IModelJson} from './base.model';

type ToolType = 'Select' | 'VERTICAL_POWER_ITERATION'; // ADD THE TOOL NAME(S) HERE
type AlgorithmType = 'Select' | 'Vertical-PCA'; // ADD THE ALGORITHM NAME(S) HERE

type StatusType = 'Created' | 'Parameters Ready' | 'Aggregating' | 'Done' | 'Aborted' | 'Failed';

export interface ProjectJson extends IModelJson {

  // Attributes common among all project types (tools)
  tool: ToolType;
  algorithm: AlgorithmType;
  name: string;
  description: string;
  status?: StatusType;
  step?: string;
  comm_round?: number;
  roles?: string[];
  token?: string;
  created_at?: string;

  center: boolean;
  scale_variance: boolean;
  log2: boolean;
  federated_qr: boolean;
  send_final_result: boolean;
  current_iteration?: number;
  max_iterations: number;
  max_dimensions: number;
  epsilon: number;

}

export class ProjectModel extends BaseModel<ProjectJson> {

  private _tool: ToolType;
  private _algorithm: AlgorithmType;
  private _name: string;
  private _description: string;
  private _status: StatusType;
  private _step: string;
  private _commRound: number;
  private _roles: string[];
  private _createdAt: Date;

  private _center: boolean;
  private _scaleVariance: boolean;
  private _log2: boolean;
  private _federated_qr: boolean;
  private _send_final_result: boolean;
  private _current_iteration: number;
  private _epsilon: number;
  private _maxDimensions: number;
  private _maxIterations: number;


  constructor() {
    super();
  }

  public async refresh(proj: ProjectJson) {
    this._id = proj.id;
    this._tool = proj.tool;
    this._algorithm = proj.algorithm;
    this._name = proj.name;
    this._description = proj.description;
    this._status = proj.status;
    this._step = proj.step;
    this._commRound = proj.comm_round;
    this._roles = proj.roles;
    this._createdAt = new Date(proj.created_at);

    this._center = proj.center;
    this._scaleVariance = proj.scale_variance;
    this._log2 = proj.log2;
    this._federated_qr = proj.federated_qr;
    this._send_final_result = proj.send_final_result;
    this._current_iteration = proj.current_iteration;
    this._maxDimensions = proj.max_dimensions;
    this._maxIterations = proj.max_iterations;
    this._epsilon = proj.epsilon;
  }

  public get tool(): ToolType {
    return this._tool;
  }

  public get algorithm(): AlgorithmType {
    return this._algorithm;
  }

  public get name(): string {
    return this._name;
  }

    public get description(): string {
    return this._description;
  }

  public get status(): StatusType {
    return this._status;
  }

  public get step(): string {
    return this._step;
  }

  public get commRound(): number {
    return this._commRound;
  }

  public get roles(): string[] {
    return this._roles;
  }

  public get createdAt(): Date {
    return this._createdAt;
  }

  public get center(): boolean {
    return this._center;
  }

  public get scale_variance(): boolean {
    return this._scaleVariance;
  }

  public get log2(): boolean {
    return this._log2;
  }

  public get federated_qr(): boolean {
    return this._federated_qr;
  }
  public get current_iteration(): number {
    return this._current_iteration;
  }

  public get send_final_result(): boolean {
    return this._send_final_result;
  }

  public get dimensions(): number {
    return this._maxDimensions;
  }

  public get maxIterations(): number {
    return this._maxIterations;
  }

  public get epsilon(): number {
    return this._epsilon;
  }

}