package gym;

import engine.core.*;
import engine.helper.Assets;
import engine.helper.GameStatus;
import engine.helper.MarioActions;
import py4j.GatewayServer;

import javax.swing.*;
import java.awt.*;
import java.awt.image.VolatileImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

public class MarioGym {
    String level;
    int gameSeconds;
    int marioState;
    //static boolean visual;

    //Visualisation
    JFrame window = null;
    MarioRender render = null;
    VolatileImage renderTarget = null;
    Graphics backBuffer = null;
    Graphics currentBuffer = null;

    //Game and character
    MarioWorld world = null;
    Py4JAgent agent = null;
    MarioTimer agentTimer = null;

    //GameLoop
    ArrayList<MarioEvent> gameEvents = null;
    ArrayList<MarioAgentEvent> agentEvents = null;

    //Step related
    int sceneDetail = 0;
    int enemyDetail = 0;

    float rewardCombined = 0.0f;
    int currentCheckpoint = 0;
    float totalReward = 0.0f;

    //boolean updateReward = false;
    int rewardFunction = 0;
    int lastRewardMark = 0;
    float lastMarioX = 0;
    int gymID = 0;

    /*
    public static void main(String[] args) {
        MarioGym gym = new MarioGym();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(gym);
        server.start();
        System.out.println("Gateway Started");
    }
    */

    private int[][][] getObservation(){
        return world.getMergedObservation3(world.mario.x, world.mario.y, 1, 1);
    }

    private boolean[] convertAgentInput(int number){
        boolean[] agentInput;
        switch (number){
            case 0:
                agentInput = new boolean[] {false, false, false, false, false};
                break;
            case 1:
                agentInput = new boolean[] {false, false, false, false, true};
                break;
            case 2:
                agentInput = new boolean[] {false, false, false, true, false};
                break;
            case 3:
                agentInput = new boolean[] {false, false, false, true, true};
                break;
            case 4:
                agentInput = new boolean[] {false, false, true, false, false};
                break;
            case 5:
                agentInput = new boolean[] {false, false, true, false, true};
                break;
            case 6:
                agentInput = new boolean[] {false, false, true, true, false};
                break;
            case 7:
                agentInput = new boolean[] {false, false, true, true, true};
                break;
            case 8:
                agentInput = new boolean[] {false, true, false, false, false};
                break;
            case 9:
                agentInput = new boolean[] {false, true, false, false, true};
                break;
            case 10:
                agentInput = new boolean[] {false, true, false, true, false};
                break;
            case 11:
                agentInput = new boolean[] {false, true, false, true, true};
                break;
            case 12:
                agentInput = new boolean[] {false, true, true, false, false};
                break;
            case 13:
                agentInput = new boolean[] {false, true, true, false, true};
                break;
            case 14:
                agentInput = new boolean[] {false, true, true, true, false};
                break;
            case 15:
                agentInput = new boolean[] {false, true, true, true, true};
                break;
            case 16:
                agentInput = new boolean[] {true, false, false, false, false};
                break;
            case 17:
                agentInput = new boolean[] {true, false, false, false, true};
                break;
            case 18:
                agentInput = new boolean[] {true, false, false, true, false};
                break;
            case 19:
                agentInput = new boolean[] {true, false, false, true, true};
                break;
            case 20:
                agentInput = new boolean[] {true, false, true, false, false};
                break;
            case 21:
                agentInput = new boolean[] {true, false, true, false, true};
                break;
            case 22:
                agentInput = new boolean[] {true, false, true, true, false};
                break;
            case 23:
                agentInput = new boolean[] {true, false, true, true, true};
                break;
            case 24:
                agentInput = new boolean[] {true, true, false, false, false};
                break;
            case 25:
                agentInput = new boolean[] {true, true, false, false, true};
                break;
            case 26:
                agentInput = new boolean[] {true, true, false, true, false};
                break;
            case 27:
                agentInput = new boolean[] {true, true, false, true, true};
                break;
            case 28:
                agentInput = new boolean[] {true, true, true, false, false};
                break;
            case 29:
                agentInput = new boolean[] {true, true, true, false, true};
                break;
            case 30:
                agentInput = new boolean[] {true, true, true, true, false};
                break;
            case 31:
                agentInput = new boolean[] {true, true, true, true, true};
                break;
            default:
                agentInput = new boolean[] {false, false, false, false, false};
        }
        return agentInput;
    }

    public StepReturnType step(int number){
        boolean[] input = convertAgentInput(number);
        agentInput(input[0],input[1],input[2],input[3],input[4]);
        gameUpdate();
        StepReturnType returnVal = new StepReturnType();
        //Done value
        if (world.gameStatus == GameStatus.RUNNING) returnVal.done = false;
        else returnVal.done = true;
        //Reward value
        returnVal.reward = rewardCombined;
        //State value
        returnVal.state = getObservation();
        //Info values
        returnVal.info = new HashMap<>();
        returnVal.marioPosition = world.mario.x;
        if(world.gameStatus == GameStatus.WIN) returnVal.info.put("Result", "Win");
        else if (world.gameStatus == GameStatus.LOSE) returnVal.info.put("Result", "Lose");
        returnVal.info.put("Yolo","Swaggins");
        totalReward += returnVal.reward;
        returnVal.info.put("ReturnScore", String.valueOf(totalReward));
        return returnVal;
    }

    public void init(int id,String paramLevel, String imageDirectory, int timer, int paramMarioState, boolean visual, int paramRewardFunction){
        gymID = id;
        rewardFunction = paramRewardFunction;
        level = paramLevel;
        gameSeconds = timer;
        marioState = paramMarioState;
        Assets.img = imageDirectory;
        sceneDetail = 1;
        enemyDetail = 1;

        if (visual) {
            window = new JFrame("Mario AI Framework");
            render = new MarioRender(2);
            window.setContentPane(render);
            window.pack();
            window.setResizable(false);
            window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            render.init();
            window.setVisible(true);
        }

        reset(visual);
        System.out.println("Gym initialised");
    }

    public float reward(int index){
        switch (index){
            case 0:
                return rewardOriginal();
            case 1:
                return reward1();
            case 2:
                return reward2();
            case 3:
                return reward3();
            case 4:
                return reward4();
            case 5:
                return reward5();
            case 6:
                return reward6();
            case 7:
                return reward7();
            case 8:
                return reward8();
            case 9:
                return reward9();
            case 10:
                return reward10();
            case 11:
                return reward11();
            default:
                return rewardOriginal();
        }
    }

    public float rewardOriginal(){
        float rewardPos = 0;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 15;
        int winMultiplier = 10;

        float newMarioX = world.mario.x;
        rewardPos = newMarioX - lastMarioX;
        lastMarioX = newMarioX;
        rewardTimePenalty = -1;

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward1(){
        float rewardPos = 0;
        float rewardPosClip = 15;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 200;
        int winMultiplier = 5;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            float newMarioX = world.mario.x;
            rewardPos = newMarioX - lastMarioX;
            rewardPos = Math.max(-rewardPosClip, Math.min(rewardPosClip, rewardPos));
            lastMarioX = newMarioX;
            rewardTimePenalty = -1;
        }
        else{
            rewardPos = 0.0f;
            rewardTimePenalty = 0;
        }

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward2(){
        //Same as original, but time penalty is scaled to be smaller, and add up to -1 over one second, and win reward is larger
        float rewardPos = 0;
        float rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 15;
        int winMultiplier = 20;

        float newMarioX = world.mario.x;
        rewardPos = newMarioX - lastMarioX;
        lastMarioX = newMarioX;
        rewardTimePenalty = ((-1f)/30f);

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward3(){
        //Same as reward1, but with smaller death penalties and smaller win reward
        float rewardPos = 0;
        float rewardPosClip = 15;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 25;
        int winMultiplier = 20;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            float newMarioX = world.mario.x;
            rewardPos = newMarioX - lastMarioX;
            rewardPos = Math.max(-rewardPosClip, Math.min(rewardPosClip, rewardPos));
            lastMarioX = newMarioX;
            rewardTimePenalty = -1;
        }
        else{
            rewardPos = 0.0f;
            rewardTimePenalty = 0;
        }

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward4(){
        //same as original, except the time penalty is only applied one time each second
        float rewardPos = 0;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 15;
        int winMultiplier = 10;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            rewardTimePenalty = -1;
        }
        else{
            rewardTimePenalty = 0;
        }

        float newMarioX = world.mario.x;
        rewardPos = newMarioX - lastMarioX;
        lastMarioX = newMarioX;

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward5(){
        //same as original, except no time penalty
        float rewardPos = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 15;
        int winMultiplier = 10;

        float newMarioX = world.mario.x;
        rewardPos = newMarioX - lastMarioX;
        lastMarioX = newMarioX;

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward6(){
        //Same as reward3, but with bigger win reward
        float rewardPos = 0;
        float rewardPosClip = 15;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 25;
        int winMultiplier = 40;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            float newMarioX = world.mario.x;
            rewardPos = newMarioX - lastMarioX;
            rewardPos = Math.max(-rewardPosClip, Math.min(rewardPosClip, rewardPos));
            lastMarioX = newMarioX;
            rewardTimePenalty = -1;
        }
        else{
            rewardPos = 0.0f;
            rewardTimePenalty = 0;
        }

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward7(){
        //Same as reward3, but with larger movement cliprange
        float rewardPos = 0;
        float rewardPosClip = 150;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 25;
        int winMultiplier = 20;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            float newMarioX = world.mario.x;
            rewardPos = newMarioX - lastMarioX;
            rewardPos = Math.max(-rewardPosClip, Math.min(rewardPosClip, rewardPos));
            lastMarioX = newMarioX;
            rewardTimePenalty = -1;
        }
        else{
            rewardPos = 0.0f;
            rewardTimePenalty = 0;
        }

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward8(){
        //Same as reward7, but with larger movement cliprange and larger win reward, cliprange reward half of compounded reward in original
        float rewardPos = 0;
        float rewardPosClip = 15*15;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 25;
        int winMultiplier = 40;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            float newMarioX = world.mario.x;
            rewardPos = newMarioX - lastMarioX;
            rewardPos = Math.max(-rewardPosClip, Math.min(rewardPosClip, rewardPos));
            lastMarioX = newMarioX;
            rewardTimePenalty = -1;
        }
        else{
            rewardPos = 0.0f;
            rewardTimePenalty = 0;
        }

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward9(){
        //Same as reward8, but with larger movement cliprange, compounded reward equal to original
        float rewardPos = 0;
        float rewardPosClip = 15*30;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 25;
        int winMultiplier = 40;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            float newMarioX = world.mario.x;
            rewardPos = newMarioX - lastMarioX;
            rewardPos = Math.max(-rewardPosClip, Math.min(rewardPosClip, rewardPos));
            lastMarioX = newMarioX;
            rewardTimePenalty = -1;
        }
        else{
            rewardPos = 0.0f;
            rewardTimePenalty = 0;
        }

        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward10(){
        //checkpoint based rewards, 10 checkpoints, large win reward, time penalty every second
        int checkPoints = 10;
        int checkPointReward = 150;
        float rewardPos = 0;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 15;
        int winMultiplier = 50;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            rewardTimePenalty = -1;
        }
        else{
            rewardTimePenalty = 0;
        }

        lastMarioX = world.mario.x;
        //Should be >= if 10 checkpoints are wanted, with only > it never hits the last checkpoint, as that last checkpoint is on the win-tile
        //It is probably fine still, as the checkpoints are still spaced with 10% of the level apart, and the last checkpoint
        //then doubles as both a checkpoint and as goal, and conceptually goal is kinda fine as a final checkpoint
        if (lastMarioX > (float)(currentCheckpoint+1)/checkPoints * world.level.exitTileX*16){
            currentCheckpoint++;
            rewardPos = checkPointReward;
        }


        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward11(){
        //checkpoint based rewards, 20 checkpoints, large win reward, time penalty every second
        int checkPoints = 20;
        int checkPointReward = 150;
        float rewardPos = 0;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 15;
        int winMultiplier = 50;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            rewardTimePenalty = -1;
        }
        else{
            rewardTimePenalty = 0;
        }

        lastMarioX = world.mario.x;
        if (lastMarioX > (float)(currentCheckpoint+1)/checkPoints * world.level.exitTileX*16){
            currentCheckpoint++;
            rewardPos = checkPointReward;
        }


        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }

    public float reward12(){
        //same as 10 but larger death penalty
        int checkPoints = 10;
        int checkPointReward = 150;
        float rewardPos = 0;
        int rewardTimePenalty = 0;
        int rewardDeathPenalty = 0;

        int winLooseReward = 750;
        int winMultiplier = 1;

        if(world.currentTick - lastRewardMark == 30){
            lastRewardMark = world.currentTick;
            rewardTimePenalty = -1;
        }
        else{
            rewardTimePenalty = 0;
        }

        lastMarioX = world.mario.x;
        //Should be >= if 10 checkpoints are wanted, with only > it never hits the last checkpoint, as that last checkpoint is on the win-tile
        //It is probably fine still, as the checkpoints are still spaced with 10% of the level apart, and the last checkpoint
        //then doubles as both a checkpoint and as goal, and conceptually goal is kinda fine as a final checkpoint
        if (lastMarioX > (float)(currentCheckpoint+1)/checkPoints * world.level.exitTileX*16){
            currentCheckpoint++;
            rewardPos = checkPointReward;
        }


        if(world.gameStatus == GameStatus.LOSE) rewardDeathPenalty = -winLooseReward;
        else if(world.gameStatus == GameStatus.WIN) rewardDeathPenalty = winLooseReward*winMultiplier;
        else rewardDeathPenalty = 0;

        float reward = rewardPos + rewardTimePenalty + rewardDeathPenalty;
        reward = Math.max(-winLooseReward, Math.min(winLooseReward*winMultiplier, reward));

        return reward;
    }


    public void gameUpdate(){
        if (world.gameStatus == GameStatus.RUNNING) {
            //System.out.println(currentTime);
            //get actions
            agentTimer = new MarioTimer(MarioGame.maxTime);
            boolean[] actions = agent.getActions(new MarioForwardModel(world.clone()), agentTimer);
            if (MarioGame.verbose) {
                if (agentTimer.getRemainingTime() < 0 && Math.abs(agentTimer.getRemainingTime()) > MarioGame.graceTime) {
                    System.out.println("The Agent is slowing down the game by: "
                            + Math.abs(agentTimer.getRemainingTime()) + " msec.");
                }
            }
            //Reward info before update
            //int tickBeforeUpdate = world.currentTick;
            //float marioXBeforeUpdate = world.mario.x;


            // update world
            world.update(actions);
            gameEvents.addAll(world.lastFrameEvents);
            agentEvents.add(new MarioAgentEvent(actions, world.mario.x,
                    world.mario.y, (world.mario.isLarge ? 1 : 0) + (world.mario.isFire ? 1 : 0),
                    world.mario.onGround, world.currentTick));

            //Reward info after update
            rewardCombined = reward(rewardFunction);
            //System.out.println("Tick:" + world.currentTick + " : Pos:" + rewardPos + " : Total:" + totalReward);
            //Calculate reward components
            //rewardPos = marioXAfterUpdate - marioXBeforeUpdate;
            //rewardTimePenalty = tickBeforeUpdate - tickAfterUpdate;
            //System.out.println("Postion reward: " + rewardPos + ", Time reward: " + rewardTimePenalty + ", Death reward: " + rewardDeathPenalty);

        }
    }

    public StepReturnType reset(boolean visual){
        boolean won = false;
        if(world != null)  won = (world.gameStatus == GameStatus.WIN);
        agent = new Py4JAgent();
        world = new MarioWorld(null);

        world.visuals = visual;
        world.initializeLevel(level, 1000 * gameSeconds);
        if (visual) {
            world.initializeVisuals(render.getGraphicsConfiguration());
        }
        world.mario.isLarge = marioState > 0;
        world.mario.isFire = marioState > 1;

        world.update(new boolean[MarioActions.numberOfActions()]);

        //initialize graphics
        renderTarget = null;
        backBuffer = null;
        currentBuffer = null;
        if (visual) {
            renderTarget = render.createVolatileImage(MarioGame.width, MarioGame.height);
            backBuffer = render.getGraphics();
            currentBuffer = renderTarget.getGraphics();
            render.addFocusListener(render); //TODO: Maybe not needed
        }

        agentTimer = new MarioTimer(MarioGame.maxTime);
        agent.initialize(new MarioForwardModel(world.clone()), agentTimer);

        gameEvents = new ArrayList<>();
        agentEvents = new ArrayList<>();

        System.out.println("Gym Reset : ID=" + gymID + " : Win=" + (won ? "W" : "F")  + " : Return=" + totalReward);
        totalReward = 0;
        currentCheckpoint = 0;
        lastRewardMark = 0;
        lastMarioX = world.mario.x;

        StepReturnType returnVal = new StepReturnType();
        returnVal.done = false;
        returnVal.reward = 0;
        returnVal.state = getObservation();
        returnVal.info = new HashMap<>();
        return returnVal;
    }

    public void render(){
        render.renderWorld(world, renderTarget, backBuffer, currentBuffer);
    }

    public void agentInput(boolean left, boolean right, boolean down, boolean speed, boolean jump){
        boolean[] actions = new boolean[]{left, right, down, speed, jump};
        agent.setActions(actions);
    }

    public void setLevel(String levelParam){
        level = levelParam;
    }
}
